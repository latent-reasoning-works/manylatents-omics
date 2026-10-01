"""Evaluate track-effect geometry against reference and transductive clouds."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, pairwise_distances

from manylatents.metrics import (
    grouped_average_precision,
    local_participation_ratio,
    lof_novelty_score,
    pca_reference_scores,
    standardize_against,
)
from manylatents.metrics.gpd_lid import (
    exponentiality,
    hill_tail_index,
    pickands_tail_index,
    tail_distances,
)
from manylatents.metrics.local_spectral_analysis import LocalSpectralAnalysis
from manylatents.metrics.outlier_score import OutlierScore
from manylatents.utils.exceptions import MeasurementUnavailable
from manylatents.utils.stats import partial_spearman
from manylatents.utils.surrogates import (
    gaussian_surrogate,
    permute_within_groups,
    shuffle_within_rows,
)

from manylatents.dogma.benchmarks.loci import (
    distant_neighbour_distances,
    genomic_separation,
    positional_scores,
)


@dataclass(frozen=True)
class Settings:
    """Analysis settings and score orientations.

    A positive sign ranks larger values first; None reports both orientations
    without choosing a primary one. Omitted signs use ``lid_sign`` for LID,
    positive signs for magnitude, distance, LOF and PCA scores, and None for
    the remaining structural scores. Supplied signs override these defaults.
    """

    primary_k: int = 20
    pickands_k: int = 100
    k_grid: tuple[int, ...] = (10, 20, 50, 100)
    pca_components: tuple[int, ...] = (10, 50)
    n_bootstrap: int = 1000
    n_permutations: int = 1000
    seed: int = 0
    lid_sign: float = -1.0
    n_control_seeds: int = 5
    locus_exclusions: tuple[float, ...] = (0, 1_000, 100_000, 1_000_000, float("inf"))
    signs: Mapping[str, float | None] | None = None

    def __post_init__(self):
        for name in ("k_grid", "pca_components", "locus_exclusions"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        signs = {
            "lid": self.lid_sign,
            "knn_distance": +1.0,
            "lof": +1.0,
            "max": +1.0,
            "l2": +1.0,
            "mean": +1.0,
            "pickands_xi": None,
            "exponentiality": None,
            "participation_ratio": None,
        }
        for n in self.pca_components:
            signs[f"pca{n}_leading"] = +1.0
            signs[f"pca{n}_residual"] = +1.0
        if self.signs is not None:
            signs.update(self.signs)
        object.__setattr__(self, "signs", MappingProxyType(signs))


def measure_defined(fn, n_points: int):
    """Evaluate a per-point measurement, dropping points where it is undefined.

    ``fn(keep)`` computes the measurement for the query rows in ``keep``. A
    MeasurementUnavailable that names points (``.indices``) removes exactly
    those and retries; one that names none makes the whole score unavailable.
    Returns (values with NaN at dropped points, number dropped, reason or None).
    """
    keep = np.arange(n_points)
    reason = "still undefined after 10 rounds of dropping points"
    for _ in range(10):
        try:
            block = np.asarray(fn(keep), dtype=np.float64)
            values = np.full((n_points,) + block.shape[1:], np.nan)
            values[keep] = block
            return values, n_points - keep.size, None
        except MeasurementUnavailable as err:
            bad = getattr(err, "indices", None)
            reason = str(err)
            if bad is None or len(bad) == 0:
                break
            keep = np.delete(keep, bad)
            if keep.size == 0:
                break
    return None, n_points, reason


def aggregation_scores(query: np.ndarray) -> dict:
    """Per-row absolute maximum, L2 norm and mean absolute effect."""
    magnitude = np.abs(query)
    return {
        "max": magnitude.max(axis=1),
        "l2": np.linalg.norm(query, axis=1),
        "mean": magnitude.mean(axis=1),
    }


def reference_scores(query: np.ndarray, reference, k: int, pickands_k: int,
                     only=None, *, settings: Settings):
    """Label-free scores of each query row. ``reference=None`` is the transductive cloud.

    Returns (scores, dropped) where dropped[name] = (n_dropped, reason).
    ``only`` selects measurement blocks; coupled distance and PCA scores are
    returned together. ``settings`` supplies the PCA component counts.
    """
    n = query.shape[0]
    cloud = query if reference is None else reference
    scores, dropped = {}, {}

    def wanted(*names):
        return only is None or bool(set(names) & set(only))

    def add(names, fn):
        """names: one score name, or a tuple when fn returns one column per name."""
        many = isinstance(names, tuple)
        values, n_dropped, reason = measure_defined(fn, n)
        for j, name in enumerate(names if many else (names,)):
            if values is None:
                scores[name] = np.full(n, np.nan)
            else:
                scores[name] = values[:, j] if many else values
            if n_dropped:
                dropped[name] = (n_dropped, reason)

    def distance_block(keep):
        d = tail_distances(query[keep], k=k, reference=reference)
        return np.column_stack([1.0 / hill_tail_index(d), exponentiality(d), d.mean(axis=1)])

    if wanted("lid", "exponentiality", "knn_distance"):
        add(("lid", "exponentiality", "knn_distance"), distance_block)
    if wanted("pickands_xi") and pickands_k < len(cloud):
        add("pickands_xi", lambda keep: pickands_tail_index(
            tail_distances(query[keep], k=pickands_k, reference=reference)))
    if reference is None:
        if wanted("participation_ratio"):
            add("participation_ratio", lambda keep: np.asarray(
                LocalSpectralAnalysis(query[keep], n_neighbors=k, return_per_sample=True)))
        if wanted("lof"):
            add("lof", lambda keep: np.asarray(
                OutlierScore(query[keep], k=k, return_scores=True)["scores"]))
    else:
        if wanted("participation_ratio"):
            add("participation_ratio", lambda keep: local_participation_ratio(query[keep], reference, k=k))
        if wanted("lof"):
            add("lof", lambda keep: lof_novelty_score(query[keep], reference, k=k))
    for n_components in settings.pca_components:
        names = (f"pca{n_components}_leading", f"pca{n_components}_residual")
        if only is not None and not set(names) & set(only):
            continue
        try:
            pca = pca_reference_scores(query, cloud, n_components)
            scores[names[0]], scores[names[1]] = pca["leading_norm"], pca["residual_norm"]
        except MeasurementUnavailable as err:
            dropped[names[0]] = dropped[names[1]] = (n, str(err))
    return scores, dropped


def transform_features(query: np.ndarray, reference, how: str):
    """Transform features, using reference moments for standardization."""
    if how == "raw":
        return query, reference
    if how == "log":
        return np.log(query), None if reference is None else np.log(reference)
    if how == "unit":
        def unit(x):
            norm = np.linalg.norm(x, axis=1, keepdims=True)
            if not np.all(norm > 0):
                raise MeasurementUnavailable("a row with zero norm has no direction")
            return x / norm
        return unit(query), None if reference is None else unit(reference)
    if how == "zscore":
        cloud = query if reference is None else reference
        query_z, cloud_z, _ = standardize_against(query, cloud, constant="drop")
        return query_z, None if reference is None else cloud_z
    raise ValueError(how)


# Evaluation

def _auprc(score, labels, groups, rng=None, n_bootstrap=0, resample="rows"):
    finite = np.isfinite(score)
    result = grouped_average_precision(
        score[finite], labels[finite], groups[finite],
        n_bootstrap=n_bootstrap, rng=rng, resample=resample,
    )
    return result["auprc"], result["se"]


def summarize_score(name, values, labels, groups, covariates):
    """AUPRC in both orientations and rank associations with magnitude and label."""
    finite = np.isfinite(values)
    out = {"n_dropped": int((~finite).sum())}
    if finite.sum() < labels.size * 0.5:
        out["unavailable"] = True
        return out
    try:
        out["auprc_plus"], _ = _auprc(values, labels, groups)
        out["auprc_minus"], _ = _auprc(-values, labels, groups)
    except MeasurementUnavailable as err:
        out["unavailable"] = str(err)
        return out
    out["spearman_with_l2"] = float(spearmanr(values[finite], covariates[finite, 0]).statistic)
    if name not in ("l2", "max"):
        try:
            partial = partial_spearman(values[finite], labels[finite].astype(float),
                                       covariates[finite])
            out["partial_rho_given_l2_max"] = partial["rho"]
            out["partial_p"] = partial["p_value"]
        except MeasurementUnavailable as err:
            out["partial_unavailable"] = str(err)
    return out


def evaluate_score(name, values, labels, groups, covariates, rng, *, settings: Settings):
    """Summarize a score and bootstrap AUPRC in its configured orientation."""
    out = summarize_score(name, values, labels, groups, covariates)
    sign = settings.signs.get(name)
    out["prefixed_sign"] = sign
    if sign is not None and "auprc_plus" in out:
        # Resample variants within chromosomes, then whole chromosomes.
        out["auprc"], out["se"] = _auprc(sign * values, labels, groups, rng=rng,
                                        n_bootstrap=settings.n_bootstrap)
        _, out["se_groups"] = _auprc(sign * values, labels, groups, rng=rng,
                                    n_bootstrap=settings.n_bootstrap, resample="groups")
    return out


def within_magnitude(values, labels, l2, sign, n_bins=5):
    """Pooled average precision inside each L2 quintile (bins from the full cohort)."""
    edges = np.quantile(l2, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, l2, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        inside = (bins == b) & np.isfinite(values)
        positives = int(labels[inside].sum())
        row = {"bin": b, "n": int(inside.sum()), "n_positive": positives}
        if 0 < positives < inside.sum():
            row["ap_plus"] = float(average_precision_score(labels[inside], values[inside]))
            row["ap_minus"] = float(average_precision_score(labels[inside], -values[inside]))
            if sign is not None:
                row["ap"] = row["ap_plus"] if sign > 0 else row["ap_minus"]
            row["ap_l2"] = float(average_precision_score(labels[inside], l2[inside]))
            row["chance"] = positives / int(inside.sum())
        rows.append(row)
    return rows


def _summarize(values: list) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


# Locus confound

def neighbour_scores(nearest: np.ndarray) -> dict:
    """LID and mean distance from (N, k) neighbour distances.

    Rows with a zero or missing distance receive NaN, reported as dropped.
    """
    usable = np.all(np.isfinite(nearest) & (nearest > 0), axis=1)
    lid = np.full(nearest.shape[0], np.nan)
    keep = np.flatnonzero(usable)
    if keep.size:
        values, _, _ = measure_defined(lambda rows: 1.0 / hill_tail_index(nearest[keep][rows]), keep.size)
        if values is not None:
            lid[keep] = values
    return {"lid": lid, "knn_distance": np.where(usable, nearest.mean(axis=1), np.nan)}


# The full analysis

def run(*, query, clouds, variants, primary_cloud, transforms, control_settings,
        random_tracks, rng, report, settings: Settings, primary_transform="raw", log=print):
    """Fill ``report`` with the full analysis of one query matrix.

    Args:
        query: (N, T) track effects of the benchmark variants.
        clouds: name -> (M, T) reference cloud, or None for the transductive cloud.
        variants: table aligned with ``query``, with chrom, pos and boolean label.
        primary_cloud: key of ``clouds`` used for the primary setting.
        transforms: feature transforms for the sensitivity grid.
        control_settings: (cloud, transform, k) triples that get the mismatched controls.
        random_tracks: ``rng -> (query, {cloud name: reference or None})`` with the
            same number of tracks drawn at random from all assays, or None.
        rng: generator for bootstraps and permutations.
        report: dict to fill; returned, preserving unrelated existing keys.
        settings: neighbour counts, resampling counts and score orientations.
        primary_transform: feature transform for the primary setting.
        log: callable receiving each progress message; defaults to print.

    Returns:
        (report, primary_scores): the per-variant scores of the primary setting.
    """
    started = time.time()
    labels = variants["label"].to_numpy()
    groups = variants["chrom"].to_numpy()
    baselines = aggregation_scores(query)
    covariates = np.column_stack([baselines["l2"], baselines["max"]])
    report.update({
        "n_variants": int(len(variants)), "n_positive": int(labels.sum()),
        "n_tracks": int(query.shape[1]),
        "clouds": {name: (int(len(variants)) if c is None else int(c.shape[0]))
                   for name, c in clouds.items()},
        "primary": {"cloud": primary_cloud, "transform": primary_transform, "k": settings.primary_k,
                    "pickands_k": settings.pickands_k, "lid_sign": settings.lid_sign},
        "prevalence": float(labels.mean()),
    })

    # 1. Primary setting: every score
    q, r = transform_features(query, clouds[primary_cloud], primary_transform)
    primary_scores, primary_dropped = reference_scores(
        q, r, settings.primary_k, settings.pickands_k, settings=settings)
    primary_scores = {**baselines, **primary_scores}
    report["dropped"] = {k: {"n": v[0], "reason": v[1]} for k, v in primary_dropped.items()}
    report["scores"] = {
        name: evaluate_score(name, values, labels, groups, covariates, rng, settings=settings)
        for name, values in primary_scores.items()
    }
    report["within_l2_quintile"] = {
        name: within_magnitude(primary_scores[name], labels, baselines["l2"], settings.signs[name])
        for name in ("lid", "knn_distance", "lof", "participation_ratio") if name in primary_scores
    }
    for name in primary_scores:
        s = report["scores"][name]
        log(f"{name:22s} +{s.get('auprc_plus', float('nan')):.4f} / -{s.get('auprc_minus', float('nan')):.4f}"
              f"  rho_l2={s.get('spearman_with_l2', float('nan')):+.2f}"
              f"  partial_rho={s.get('partial_rho_given_l2_max', float('nan')):+.3f}"
              f"  dropped={s['n_dropped']}")

    # 2. Label-permutation null (labels permuted within chromosome)
    # The chromosome-weighted AUPRC of an uninformative score sits above the
    # prevalence, so this null, not the prevalence, is the chance level.
    lid = settings.lid_sign * primary_scores["lid"]
    finite = np.isfinite(lid)
    if finite.sum() >= labels.size * 0.5:
        null = np.array([
            grouped_average_precision(
                lid[finite], permute_within_groups(labels[finite], groups[finite], rng),
                groups[finite])["auprc"]
            for _ in range(settings.n_permutations)
        ])
        observed = report["scores"]["lid"]["auprc"]
        report["permutation_null"] = {
            "score": "lid", "mean": float(null.mean()), "sd": float(null.std(ddof=1)),
            "q95": float(np.quantile(null, 0.95)), "q99": float(np.quantile(null, 0.99)),
            "lid_p_upper": float((1 + (null >= observed).sum()) / (1 + null.size)),
        }

    # 3. Sensitivity: cloud, transform, k
    k_dependent = ("lid", "exponentiality", "knn_distance", "participation_ratio", "lof")
    grid = {}
    for cloud_name, cloud in clouds.items():
        for how in transforms:
            try:
                q, r = transform_features(query, cloud, how)
            except MeasurementUnavailable as err:
                grid[f"{cloud_name}|{how}"] = {"unavailable": str(err)}
                continue
            for k in settings.k_grid:
                only = None if k == settings.primary_k else k_dependent
                scores, _ = reference_scores(q, r, k, settings.pickands_k, only=only, settings=settings)
                grid[f"{cloud_name}|{how}|k={k}"] = {
                    name: summarize_score(name, values, labels, groups, covariates)
                    for name, values in scores.items()
                }
        log(f"sensitivity grid: {cloud_name} done ({time.time() - started:.0f} s)")
    report["sensitivity"] = grid

    # 4. Mismatched controls
    control_names = ("lid", "knn_distance", "lof", "participation_ratio",
                     "pca10_residual", "pca50_residual")
    kinds = ("row_shuffle", "gaussian_reference") + (("random_tracks",) if random_tracks else ())
    report["controls"] = {}
    for cloud_name, how, k in control_settings:
        collected = {name: {kind: {"plus": [], "minus": []} for kind in kinds}
                     for name in control_names}

        def collect(kind, scores):
            for name in control_names:
                values = scores.get(name)
                ok = values is not None and np.isfinite(values).sum() >= labels.size * 0.5
                collected[name][kind]["plus"].append(_auprc(values, labels, groups)[0] if ok else None)
                collected[name][kind]["minus"].append(_auprc(-values, labels, groups)[0] if ok else None)

        def scores_for(query_raw, reference_raw, gaussian=None):
            try:
                q, r = transform_features(query_raw, reference_raw, how)
            except MeasurementUnavailable:
                return {}
            if gaussian is not None:
                r = gaussian_surrogate(q if r is None else r, gaussian)
            return reference_scores(q, r, k, settings.pickands_k, only=control_names, settings=settings)[0]

        reference = clouds[cloud_name]
        for seed in range(settings.n_control_seeds):
            local = np.random.default_rng([settings.seed, seed])
            # (a) shuffle each variant's track values: per-row magnitudes kept, structure destroyed
            collect("row_shuffle", scores_for(
                shuffle_within_rows(query, local),
                None if reference is None else shuffle_within_rows(reference, local)))
            # (b) cloud replaced by a Gaussian with the same mean and covariance
            collect("gaussian_reference", scores_for(query, reference, gaussian=local))
            # (c) the same number of tracks drawn at random from all assays
            if random_tracks:
                random_query, random_clouds = random_tracks(local)
                collect("random_tracks", scores_for(random_query, random_clouds[cloud_name]))
        report["controls"][f"{cloud_name}|{how}|k={k}"] = {
            name: {kind: {sign: _summarize(values) for sign, values in signs.items()}
                   for kind, signs in by_kind.items()}
            for name, by_kind in collected.items()
        }
        log(f"controls: {cloud_name}|{how}|k={k} done ({time.time() - started:.0f} s)")

    # 5. Locus confound
    # Positives of a curated benchmark can sit within a few bp of one another.
    # In a transductive cloud they then find each other as neighbours, and a
    # density-based score rewards locus membership instead of anything about
    # the variant. Quantify with model-free positional scores, and by forbidding
    # nearby variants as neighbours.
    separation = genomic_separation(variants)
    nearest_positive = separation[:, labels].min(axis=1)
    report["locus"] = {
        "median_bp_to_nearest_positive": {
            "positives": float(np.median(nearest_positive[labels])),
            "controls": float(np.median(nearest_positive[~labels])),
        },
        "positional_scores": {
            name: summarize_score(name, values, labels, groups, covariates)
            for name, values in positional_scores(variants).items()
        },
        "excluded": {},
    }
    for how in ("raw", "unit"):
        try:
            x, _ = transform_features(query, None, how)
        except MeasurementUnavailable as err:
            report["locus"]["excluded"][how] = {"unavailable": str(err)}
            continue
        distance = pairwise_distances(x)
        np.fill_diagonal(distance, np.inf)
        for exclusion in settings.locus_exclusions:
            nearest = distant_neighbour_distances(distance, separation, exclusion, settings.primary_k)
            for k in (10, settings.primary_k):
                report["locus"]["excluded"][f"{how}|exclude<={exclusion:g}bp|k={k}"] = {
                    name: summarize_score(name, values, labels, groups, covariates)
                    for name, values in neighbour_scores(nearest[:, :k]).items()
                }
        del distance
    log(f"locus analysis done ({time.time() - started:.0f} s)")

    report["seconds"] = round(time.time() - started, 1)
    return report, primary_scores
