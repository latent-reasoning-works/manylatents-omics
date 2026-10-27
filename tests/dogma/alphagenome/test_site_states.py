"""The site-states command on a tiny FASTA with a fake model."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("haiku")
pytest.importorskip("pyarrow")
pytest.importorskip("pyfaidx")

from manylatents.dogma.alphagenome import model as model_module  # noqa: E402
from manylatents.dogma.alphagenome import site_states as cli  # noqa: E402
from manylatents.dogma.latent_features import cosine, delta_norm, pr3  # noqa: E402
from tests.dogma.alphagenome.fake_model import LENGTH, make_model  # noqa: E402

LAYERS = ["dna_embedder", "mha_block_0_input", "transformer_tower", "output_embedder_1"]


@pytest.fixture
def genome(tmp_path):
    rng = np.random.default_rng(0)
    sequence = "".join(rng.choice(list("ACGT"), 2000))
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + "\n".join(sequence[i:i + 60] for i in range(0, 2000, 60)) + "\n")
    return fasta, sequence


@pytest.fixture
def variants(tmp_path, genome):
    _, sequence = genome
    rows = []
    for pos in (300, 400, 500, 600, 700):
        ref = sequence[pos - 1]
        rows.append({"chrom": "1", "pos": pos, "ref": ref, "alt": "A" if ref != "A" else "C",
                     "label": 1})
    wrong = sequence[799]
    rows.append({"chrom": "1", "pos": 800, "ref": "A" if wrong != "A" else "C", "alt": "G" if wrong != "G" else "T", "label": 0})
    rows.append({"chrom": "1", "pos": 10, "ref": sequence[9], "alt": "N", "label": 0})
    rows.append({"chrom": "1", "pos": 20, "ref": sequence[19], "alt": "A" if sequence[19] != "A" else "C", "label": 0})
    rows.append({"chrom": "2", "pos": 500, "ref": "A", "alt": "C", "label": 0})
    path = tmp_path / "variants.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def fake(monkeypatch):
    calls = []
    model = make_model()
    monkeypatch.setattr(model_module, "SUPPORTED_LENGTHS", frozenset({LENGTH}))

    def load_local(**kwargs):
        calls.append(kwargs)
        return model

    monkeypatch.setattr(cli, "load_local", load_local)
    real = cli.score_variants
    scored = []

    def counting(model, variants, *args, **kwargs):
        scored.extend(variants["variant_id"])
        return real(model, variants, *args, **kwargs)

    monkeypatch.setattr(cli, "score_variants", counting)
    return model, calls, scored


def _run(tmp_path, genome, variants, *extra):
    fasta, _ = genome
    return cli.main(["--variants", str(variants), "--fasta", str(fasta), "--out", str(tmp_path / "out"),
                     "--sequence-length", str(LENGTH), "--layers", *LAYERS, "--chunk-size", "2",
                     *extra])


def test_cli_scores_resumes_and_records_skips(tmp_path, genome, variants, fake):
    _, calls, scored = fake
    assert _run(tmp_path, genome, variants, "--batch-size", "3") == 0
    shard = tmp_path / "out" / "shard_000_of_001"
    skipped = pd.read_csv(shard / "skipped.tsv", sep="\t")
    assert dict(zip(skipped["pos"], skipped["reason"])) == {
        800: "reference_mismatch", 10: "not_snv", 20: "window_off_chromosome",
        500: "missing_chromosome"}
    _, sequence = genome
    assert skipped.loc[skipped["pos"] == 800, "fasta_base"].item() == sequence[799]
    assert "label" not in skipped.columns
    assert len(scored) == 5

    states = cli.read_site_table(tmp_path / "out", "states")
    assert len(states) == 5 * len(LAYERS) * 4
    assert set(states["allele"]) == {"ref", "alt", "alt2", "alt3"}
    assert states["vector"].iloc[0].dtype == np.float16
    for (vid, layer), group in states.groupby(["variant_id", "layer"]):
        assert sorted(group["base"]) == list("ACGT")
    features = cli.read_site_table(tmp_path / "out", "features")
    deltas = cli.read_site_table(tmp_path / "out", "deltas")
    assert len(features) == len(deltas) == 5 * len(LAYERS)
    assert deltas["vector"].iloc[0].dtype == np.float32
    assert features["pr3"].between(1, 2).all()
    timing = json.loads((shard / "timing.json").read_text())
    assert timing["chunks_scored"] == 3 and timing["n_scored_this_run"] == 5
    assert timing["site_index"] == LENGTH // 2 - 1
    assert len(calls) == 1

    # A second run scores nothing new.
    scored.clear()
    assert _run(tmp_path, genome, variants, "--batch-size", "3") == 0
    assert scored == []
    timing = json.loads((shard / "timing.json").read_text())
    assert timing["chunks_scored"] == 0 and timing["chunks_resumed"] == 3
    pd.testing.assert_frame_equal(cli.read_site_table(tmp_path / "out", "features"), features)

    # A run with a different request in the same folder is refused.
    with pytest.raises(ValueError, match="window"):
        _run(tmp_path, genome, variants, "--window", "1")


def test_features_and_deltas_come_from_float32_states(tmp_path, genome, variants, fake):
    model, _, _ = fake
    assert _run(tmp_path, genome, variants) == 0
    fasta, _ = genome
    import pyfaidx

    table = cli.read_variants(variants)
    with pyfaidx.Fasta(str(fasta)) as handle:
        kept, _ = cli.prepare_variants(table, handle, LENGTH)
        states, failed = cli.score_variants(model, kept.iloc[:1], handle, LENGTH, LAYERS)
        row = kept.iloc[0]
        sites = model_module.site_sequences(row.chromosome, row.pos, row.ref, row.alt, handle, LENGTH)
    assert failed == []
    vid = kept["variant_id"].iloc[0]
    from manylatents.dogma.alphagenome.taps import site_states

    for allele in ("ref", "alt3"):
        direct = site_states(model, model_module.one_hot(sites.sequences[allele])[None],
                             sites.site_index, LAYERS)
        for layer in LAYERS:
            np.testing.assert_array_equal(states[vid][(allele, 0)][layer], direct[layer][0])
    h = {allele: states[vid][(allele, 0)]["transformer_tower"]
         for allele in ("ref", "alt", "alt2", "alt3")}
    features = cli.read_site_table(tmp_path / "out", "features", layer="transformer_tower")
    row = features[features["variant_id"] == vid].iloc[0]
    np.testing.assert_allclose(row["delta_norm"], delta_norm(h["ref"], h["alt"]), rtol=1e-6)
    np.testing.assert_allclose(row["cosine"], cosine(h["ref"], h["alt"]), rtol=1e-6)
    np.testing.assert_allclose(row["pr3"], pr3(h["ref"], h["alt"], h["alt2"], h["alt3"]), rtol=1e-6)
    matrix = cli.delta_matrix(tmp_path / "out", "transformer_tower", [vid])
    np.testing.assert_array_equal(matrix[0], h["alt"] - h["ref"])
    with pytest.raises(KeyError):
        cli.delta_matrix(tmp_path / "out", "transformer_tower", ["chr9:1:A>C"])


def test_repeat_and_ref_alt(tmp_path, genome, variants, fake):
    assert _run(tmp_path, genome, variants, "--repeat", "3", "--alleles", "ref_alt") == 0
    states = cli.read_site_table(tmp_path / "out", "states")
    assert len(states) == 5 * len(LAYERS) * 2 * 3
    assert set(states["repeat"]) == {0, 1, 2} and set(states["allele"]) == {"ref", "alt"}
    features = cli.read_site_table(tmp_path / "out", "features")
    assert len(features) == 5 * len(LAYERS) * 3 and features["pr3"].isna().all()
    # The same input gives the same state in every repeat on this device.
    for _, group in states.groupby(["variant_id", "layer", "allele"]):
        vectors = np.stack(group.sort_values("repeat")["vector"].to_list())
        np.testing.assert_array_equal(vectors[0], vectors[1])


def test_out_of_memory_is_recorded_and_retried(tmp_path, genome, variants, fake, monkeypatch):
    from manylatents.dogma.alphagenome import taps

    real = taps.site_states
    state = {"fail": True}

    class JaxRuntimeError(RuntimeError):
        pass

    def flaky(model, onehot, *args, **kwargs):
        if state["fail"]:
            raise JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate")
        return real(model, onehot, *args, **kwargs)

    monkeypatch.setattr(taps, "site_states", flaky)
    assert _run(tmp_path, genome, variants) == 0
    shard = tmp_path / "out" / "shard_000_of_001"
    assert len(pd.read_csv(shard / "scoring_failed.tsv", sep="\t")) == 5
    with pytest.raises(FileNotFoundError):
        cli.read_site_table(tmp_path / "out", "states")
    state["fail"] = False
    assert _run(tmp_path, genome, variants) == 0  # resumes: failures stay recorded
    assert len(pd.read_csv(shard / "scoring_failed.tsv", sep="\t")) == 5
    assert _run(tmp_path, genome, variants, "--retry-failed") == 0
    assert pd.read_csv(shard / "scoring_failed.tsv", sep="\t").empty
    assert len(cli.read_site_table(tmp_path / "out", "deltas")) == 5 * len(LAYERS)


def test_other_errors_propagate(tmp_path, genome, variants, fake, monkeypatch):
    from manylatents.dogma.alphagenome import taps

    def broken(*args, **kwargs):
        raise KeyError("modules not reached")

    monkeypatch.setattr(taps, "site_states", broken)
    with pytest.raises(KeyError):
        _run(tmp_path, genome, variants)


def test_shards_and_limit(tmp_path, genome, variants, fake):
    _, _, scored = fake
    for shard in range(2):
        assert _run(tmp_path, genome, variants, "--n-shards", "2", "--shard", str(shard)) == 0
    assert len(scored) == 5
    assert len(cli.shard_dirs(tmp_path / "out")) == 2
    assert len(cli.read_site_table(tmp_path / "out", "deltas", layer="dna_embedder")) == 5


def test_unsupported_length_fails_before_loading(tmp_path, genome, variants, fake):
    _, calls, _ = fake
    fasta, _ = genome
    with pytest.raises(SystemExit):
        cli.main(["--variants", str(variants), "--fasta", str(fasta), "--out", str(tmp_path / "o"),
                  "--sequence-length", "1000"])
    assert calls == []


def test_help_without_heavy_imports():
    code = '''
import sys
class BlockHeavy:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"jax", "haiku", "torch", "alphagenome", "alphagenome_research", "huggingface_hub"}:
            raise AssertionError("unexpected heavy import: " + fullname)
sys.meta_path.insert(0, BlockHeavy())
from manylatents.dogma.alphagenome.site_states import main
import manylatents.dogma.alphagenome.model, manylatents.dogma.alphagenome.taps
try:
    main(["--help"])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError("help did not exit")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--repeat REPEAT" in result.stdout


def test_states_that_overflow_float16_are_stored_as_float32():
    vid = "chr1:1:A>C"
    big = {("ref", 0): {"L": np.full(4, 7e4, np.float32), "M": np.ones(4, np.float32)},
           ("alt", 0): {"L": np.full(4, 7.1e4, np.float32), "M": np.ones(4, np.float32)}}
    info = {}
    tables = cli.chunk_tables({vid: big}, {vid: {"ref": "A", "alt": "C"}}, ["L", "M"],
                              "ref_alt", 1, info=info)
    vectors = tables["states"].column("vector").to_pylist()
    assert np.isfinite(vectors).all()
    assert vectors[0][0] == pytest.approx(7e4)
    assert info["states_dtype"] == "float32" and info["float32_layers"] == ["L"]
    assert info["max_abs"] == {"L": pytest.approx(7.1e4), "M": 1.0}
    small = {k: {"L": v["M"], "M": v["M"]} for k, v in big.items()}
    info = {}
    tables = cli.chunk_tables({vid: small}, {vid: {"ref": "A", "alt": "C"}}, ["L", "M"],
                              "ref_alt", 1, info=info)
    assert tables["states"].column("vector").type.value_type == "halffloat"
    assert info["states_dtype"] == "float16" and info["float32_layers"] == []


def test_run_records_float32_chunks_and_state_ranges(tmp_path, genome, variants, fake,
                                                     monkeypatch):
    real = cli.score_variants

    def huge_first_chunk(model, chunk, *args, **kwargs):
        states, failed = real(model, chunk, *args, **kwargs)
        first = chunk["variant_id"].iloc[0]
        if ":300:" in first:  # the first chunk
            for h in states[first].values():
                h["transformer_tower"] = h["transformer_tower"] + 7e4
        return states, failed

    monkeypatch.setattr(cli, "score_variants", huge_first_chunk)
    assert _run(tmp_path, genome, variants) == 0
    shard = tmp_path / "out" / "shard_000_of_001"
    manifest = json.loads((shard / "manifest.json").read_text())
    assert manifest["chunks"]["0"]["states_dtype"] == "float32"
    assert manifest["chunks"]["0"]["float32_layers"] == ["transformer_tower"]
    assert manifest["chunks"]["1"]["states_dtype"] == "float16"
    timing = json.loads((shard / "timing.json").read_text())
    assert timing["float32_state_chunks"] == {"0": ["transformer_tower"]}
    assert timing["max_abs_state"]["transformer_tower"] > 7e4
    assert set(timing["max_abs_state"]) == set(LAYERS)
    states = cli.read_site_table(tmp_path / "out", "states", "transformer_tower")
    assert all(np.isfinite(v).all() for v in states["vector"])
