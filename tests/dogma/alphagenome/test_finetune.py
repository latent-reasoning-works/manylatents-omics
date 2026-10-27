"""AlphaGenomeFinetune: the LatentModule over interventions, objectives and the JAX trainer."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("haiku")
pytest.importorskip("optax")
pytest.importorskip("pyarrow")
pytest.importorskip("pyfaidx")

import jax  # noqa: E402

from manylatents.dogma.alphagenome import finetune as finetune_module  # noqa: E402
from manylatents.dogma.alphagenome.finetune import AlphaGenomeFinetune, load_rows  # noqa: E402
from manylatents.dogma.alphagenome.interventions import HeadOnly, LowRankEdit  # noqa: E402
from manylatents.dogma.alphagenome.objectives import (  # noqa: E402
    Composite, DiffusionOperatorMatch, JacobianPenalty, TrackHead,
)
from manylatents.dogma.alphagenome.taps import site_states  # noqa: E402
from manylatents.dogma.alphagenome.train import JaxTrainer  # noqa: E402
from manylatents.dogma.variants import with_variant_ids  # noqa: E402
from tests.dogma.alphagenome.fake_model import LENGTH, make_model  # noqa: E402

F32 = "params=float32,compute=float32,output=float32"
POSITIONS = [300, 380, 450, 520, 600, 700, 820, 900]


class VariantsModule:
    def __init__(self, table):
        self.table = table

    def get_variants(self):
        return self.table


@pytest.fixture(scope="module")
def model():
    return make_model(policy=F32)


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(1)
    sequence = "".join(rng.choice(list("ACGT"), 1500))
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + sequence + "\n")
    table = pd.DataFrame({"chrom": ["1"] * len(POSITIONS), "pos": POSITIONS,
                          "ref": [sequence[p - 1] for p in POSITIONS]})
    table["alt"] = ["A" if r != "A" else "G" for r in table["ref"]]
    table["label"] = [0, 1] * (len(POSITIONS) // 2)
    return fasta, table, VariantsModule(table)


def _deltas(model, fasta, table, layer):
    """Unedited alt - ref site states at ``layer``, straight from taps.site_states."""
    import pyfaidx

    from manylatents.dogma.alphagenome.model import one_hot, site_sequences

    rows = []
    with pyfaidx.Fasta(str(fasta)) as genome:
        for row in with_variant_ids(table).itertuples(index=False):
            sites = site_sequences(row.chromosome, row.pos, row.ref, row.alt, genome, LENGTH,
                                   "ref_alt")
            onehot = np.stack([one_hot(sites.sequences["ref"]), one_hot(sites.sequences["alt"])])
            states = site_states(model, onehot, sites.site_index, [layer])[layer]
            rows.append(states[1] - states[0])
    return np.stack(rows)


def _targets(tmp_path, model, fasta, table, layer="transformer_tower", seed=0):
    """A toy target: a fixed linear map of the unedited delta, scaled to unit variance."""
    delta = _deltas(model, fasta, table, layer)
    target = delta @ np.random.default_rng(seed).normal(size=(delta.shape[1], 3))
    target = (target / target.std()).astype(np.float32)
    path = tmp_path / "targets.tsv"
    frame = table[["chrom", "pos", "ref", "alt"]].copy()
    for j in range(target.shape[1]):
        frame[f"t{j}"] = target[:, j]
    frame.to_csv(path, sep="\t", index=False)
    return path, target


def _module(model, fasta, datamodule, **kwargs):
    defaults = dict(
        intervention=LowRankEdit(rank=2), layer="mha_block_4_input",
        readout_layers=["transformer_tower"],
        objective=TrackHead(layers="transformer_tower", source="delta"),
        trainer=JaxTrainer(lr=1e-2, max_steps=30, batch_size=4, log_every=5),
        fasta=str(fasta), sequence_length=LENGTH, model=model, datamodule=datamodule)
    defaults.update(kwargs)
    return AlphaGenomeFinetune(**defaults)


def test_fit_transform_trains_only_the_edit(setup, tmp_path, model):
    fasta, table, datamodule = setup
    targets, _ = _targets(tmp_path, model, fasta, table)
    before = jax.tree_util.tree_map(np.array, model.params)
    module = _module(model, fasta, datamodule, targets_path=str(targets),
                     transform_output="delta")
    out = module.fit_transform(np.zeros((len(POSITIONS), 1)))
    assert out.shape == (len(POSITIONS), 16) and out.dtype == np.float32
    assert np.isfinite(out).all()
    # every AlphaGenome parameter is bit-identical after training
    for a, b in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(model.params)):
        assert np.array_equal(a, np.asarray(b))
    losses = [r["loss"] for r in module.history]
    assert losses[-1] < losses[0]
    r = module.intervention.rotation(module.params["intervention"])
    np.testing.assert_allclose(np.asarray(r @ r.T), np.eye(2), atol=1e-5)
    # the trained edit changes the representation
    unedited = _deltas(model, fasta, table, "transformer_tower")
    assert not np.allclose(out, unedited, rtol=1e-3)
    head = _module(model, fasta, datamodule, targets_path=str(targets),
                   transform_output="head_output")
    head.params = module.params
    assert head.transform(None).shape == (len(POSITIONS), 3)
    state = _module(model, fasta, datamodule, targets_path=str(targets))
    state.params = module.params
    assert state.transform(None).shape == (len(POSITIONS), 16)
    saved = module.save(tmp_path / "trained.npz")
    assert saved.exists()


def test_untrained_edit_transform_is_the_plain_delta(setup, tmp_path, model):
    fasta, table, datamodule = setup
    targets, _ = _targets(tmp_path, model, fasta, table)
    module = _module(model, fasta, datamodule, targets_path=str(targets),
                     transform_output="delta",
                     trainer=JaxTrainer(lr=0.0, weight_decay=0.0, max_steps=1, batch_size=4))
    module.fit(None)
    unedited = _deltas(model, fasta, table, "transformer_tower")
    np.testing.assert_allclose(module.transform(None), unedited, rtol=1e-3,
                               atol=1e-4 * np.abs(unedited).max())


def test_head_only_on_cached_states(setup, tmp_path, model):
    fasta, table, datamodule = setup
    targets, target = _targets(tmp_path, model, fasta, table)
    layers = ["mha_block_4_input", "transformer_tower"]
    module = _module(model, fasta, datamodule, intervention=HeadOnly(), readout_layers=layers,
                     objective=TrackHead(layers=layers, source="delta"),
                     trainer=JaxTrainer(lr=0.05, max_steps=600, batch_size=8, log_every=50),
                     targets_path=str(targets), transform_output="head_output")
    out = module.fit_transform(None)
    assert out.shape == target.shape
    # unit-variance targets that are a linear map of one readout: the head fits them
    assert module.history[-1]["loss"] < 0.05 * np.var(target)
    assert module.params["intervention"] == {}
    states = _module(model, fasta, datamodule, intervention=HeadOnly(), readout_layers=layers,
                     objective=TrackHead(layers=layers, source="delta"), targets_path=str(targets))
    states.params = module.params
    assert states.transform(None).shape == (len(POSITIONS), 32)


def test_operator_match_with_jacobian_penalty(setup, tmp_path, model):
    fasta, table, datamodule = setup
    teacher = tmp_path / "teacher.npy"
    np.save(teacher, np.random.default_rng(0).normal(size=(len(POSITIONS), 5)))
    objective = Composite([DiffusionOperatorMatch(layers="transformer_tower"),
                           JacobianPenalty(residual=True, weight=0.01)])
    module = _module(model, fasta, datamodule, objective=objective, teacher_path=str(teacher),
                     trainer=JaxTrainer(lr=1e-2, max_steps=10, batch_size=8, log_every=5,
                                        eval_every=5),
                     val_fraction=0.25, transform_output="delta")
    module.fit(None)
    record = module.history[-1]
    assert {"diffusion_operator_match", "jacobian_penalty", "eval/loss"} <= set(record)
    assert np.isfinite(record["loss"])
    assert module.transform(None).shape == (len(POSITIONS), 16)


def test_targets_never_come_from_y_by_default(setup, model):
    fasta, table, datamodule = setup
    y = np.asarray(table["label"], np.float32)
    module = _module(model, fasta, datamodule)
    with pytest.raises(ValueError, match="targets_from='y'"):
        module.fit(np.zeros((len(POSITIONS), 1)), y)
    allowed = _module(model, fasta, datamodule, targets_from="y",
                      trainer=JaxTrainer(max_steps=2, batch_size=4))
    allowed.fit(None, y)
    assert allowed.params["objective"]["track_head"]["layer_0"]["w"].shape == (16, 1)
    with pytest.raises(ValueError, match="no y"):
        _module(model, fasta, datamodule, targets_from="y").fit(None, None)


def test_teacher_is_required(setup, model):
    fasta, _, datamodule = setup
    with pytest.raises(ValueError, match="teacher_path"):
        _module(model, fasta, datamodule, objective=DiffusionOperatorMatch()).fit(None)


def test_bad_arguments(setup, model):
    fasta, _, datamodule = setup
    with pytest.raises(TypeError, match="unexpected keyword"):
        _module(model, fasta, datamodule, layr="transformer_tower")
    with pytest.raises(ValueError, match="unknown"):
        _module(model, fasta, datamodule, layer="nope")
    with pytest.raises(ValueError, match="readout_layers"):
        _module(model, fasta, datamodule, objective=TrackHead(layers="dna_embedder"))
    with pytest.raises(ValueError, match="JacobianPenalty"):
        _module(model, fasta, datamodule, intervention=HeadOnly(),
                objective=[TrackHead(), JacobianPenalty()])
    with pytest.raises(ValueError, match="TrackHead"):
        _module(model, fasta, datamodule, objective=DiffusionOperatorMatch(),
                transform_output="head_output")
    with pytest.raises(ValueError, match="targets_from"):
        _module(model, fasta, datamodule, targets_from="labels")
    with pytest.raises(ValueError, match="objective"):
        _module(model, fasta, datamodule, objective=None)
    with pytest.raises(RuntimeError, match="fit"):
        _module(model, fasta, datamodule).transform(None)
    with pytest.raises(ValueError, match="sequence length"):
        _module(model, fasta, datamodule, sequence_length=1024, targets_from="y").fit(
            None, np.zeros(len(POSITIONS)))


def test_load_rows(tmp_path, setup):
    _, table, _ = setup
    variants = with_variant_ids(table)
    values = np.arange(len(table) * 2, dtype=np.float32).reshape(-1, 2)
    np.save(tmp_path / "rows.npy", values)
    np.testing.assert_array_equal(load_rows(tmp_path / "rows.npy", variants), values)
    np.savez(tmp_path / "rows.npz", variant_id=variants["variant_id"].to_numpy(str)[::-1],
             values=values[::-1])
    np.testing.assert_array_equal(load_rows(tmp_path / "rows.npz", variants), values)
    frame = variants[["variant_id"]].assign(a=values[:, 0], b=values[:, 1], note="x")
    frame.iloc[::-1].to_parquet(tmp_path / "rows.parquet")
    np.testing.assert_array_equal(load_rows(tmp_path / "rows.parquet", variants), values)
    with pytest.raises(KeyError, match="no targets"):
        load_rows(tmp_path / "rows.parquet", with_variant_ids(
            pd.DataFrame({"chrom": ["2"], "pos": [5], "ref": ["A"], "alt": ["C"]})))
    with pytest.raises(ValueError, match="rows"):
        load_rows(tmp_path / "rows.npy", variants.iloc[:3])


CONFIGS = ["alphagenome_finetune", "alphagenome_finetune_operator", "alphagenome_head_only"]


@pytest.mark.parametrize("name", CONFIGS)
def test_hydra_configs_instantiate(name):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    path = (Path(finetune_module.__file__).parents[1] / "configs" / "algorithms" / "latent"
            / f"{name}.yaml")
    cfg = OmegaConf.load(path)
    cfg.fasta = "unused.fa"
    module = instantiate(cfg)
    assert isinstance(module, AlphaGenomeFinetune)
    assert isinstance(module.trainer, JaxTrainer)
    assert module.sequence_length == 131_072
    if name == "alphagenome_finetune_operator":
        assert [type(o) for o in module.objective.objectives] == [DiffusionOperatorMatch,
                                                                 JacobianPenalty]
        assert module.objective.objectives[1].weight == 0.01
    elif name == "alphagenome_head_only":
        assert isinstance(module.intervention, HeadOnly)
        assert module.readout_layers == ["mha_block_4_input", "transformer_tower"]
    else:
        assert isinstance(module.intervention, LowRankEdit)
        assert module.intervention.rank == 8
    # nested configs passed un-instantiated are built by the module
    lazy = instantiate(cfg, _recursive_=False)
    assert isinstance(lazy.trainer, JaxTrainer) and isinstance(lazy.objective, Composite)
