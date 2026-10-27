"""AlphaGenomeSiteEncoder: the LatentModule interface over site states."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("haiku")
pytest.importorskip("pyarrow")
pytest.importorskip("pyfaidx")

from manylatents.dogma.alphagenome import encoder as encoder_module  # noqa: E402
from manylatents.dogma.alphagenome import model as model_module  # noqa: E402
from manylatents.dogma.alphagenome import site_states as cli  # noqa: E402
from manylatents.dogma.alphagenome.encoder import AlphaGenomeSiteEncoder  # noqa: E402
from tests.dogma.alphagenome.fake_model import LENGTH, make_model  # noqa: E402


class VariantsModule:
    def __init__(self, table):
        self.table = table

    def get_variants(self):
        return self.table


@pytest.fixture
def setup(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    sequence = "".join(rng.choice(list("ACGT"), 1500))
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + sequence + "\n")
    table = pd.DataFrame({
        "chrom": ["1"] * 4, "pos": [300, 450, 600, 900],
        "ref": [sequence[p - 1] for p in (300, 450, 600, 900)],
    })
    table["alt"] = ["A" if r != "A" else "G" for r in table["ref"]]
    table["label"] = [0, 1, 0, 1]
    model = make_model()
    monkeypatch.setattr(model_module, "SUPPORTED_LENGTHS", frozenset({LENGTH}))
    monkeypatch.setattr(model_module, "load_local", lambda **kwargs: model)
    monkeypatch.setattr(cli, "load_local", lambda **kwargs: model)
    return fasta, table, VariantsModule(table)


def test_unknown_kwarg_and_bad_options():
    with pytest.raises(TypeError, match="unexpected keyword"):
        AlphaGenomeSiteEncoder(fasta="g.fa", layr="transformer_tower")
    with pytest.raises(ValueError, match="unknown layer"):
        AlphaGenomeSiteEncoder(fasta="g.fa", layer="nope")
    with pytest.raises(ValueError, match="output"):
        AlphaGenomeSiteEncoder(fasta="g.fa", output="tracks")
    with pytest.raises(ValueError, match="feature_layers"):
        AlphaGenomeSiteEncoder(fasta="g.fa", output="features", feature_layers=["nope"])
    with pytest.raises(ValueError, match="fasta"):
        AlphaGenomeSiteEncoder()


def test_requires_datamodule_with_variants(setup):
    fasta, _, _ = setup
    with pytest.raises(ValueError, match="datamodule"):
        AlphaGenomeSiteEncoder(fasta=str(fasta)).transform(None)
    with pytest.raises(ValueError, match="get_variants"):
        AlphaGenomeSiteEncoder(fasta=str(fasta), datamodule=object()).transform(None)


def test_delta_from_the_model(setup):
    fasta, table, datamodule = setup
    enc = AlphaGenomeSiteEncoder(layer="mha_block_3_input", fasta=str(fasta),
                                 sequence_length=LENGTH, datamodule=datamodule, batch_size=2)
    assert enc.fit(np.arange(4.0)[:, None]) is None
    out = enc.fit_transform(np.arange(4.0)[:, None])
    assert out.shape == (4, 16) and out.dtype == np.float32
    assert np.isfinite(out).all() and np.abs(out).sum() > 0


def test_features_from_the_model(setup):
    fasta, _, datamodule = setup
    layers = ["dna_embedder", "transformer_tower"]
    enc = AlphaGenomeSiteEncoder(output="features", feature_layers=layers, fasta=str(fasta),
                                 sequence_length=LENGTH, datamodule=datamodule)
    out = enc.transform(None)
    assert out.shape == (4, 6)
    assert enc.feature_names[:3] == ["dna_embedder/delta_norm", "dna_embedder/cosine",
                                     "dna_embedder/pr3"]
    assert ((out[:, 2::3] >= 1) & (out[:, 2::3] <= 2)).all()


def test_unscorable_variant_raises_instead_of_dropping_a_row(setup):
    fasta, table, _ = setup
    bad = table.copy()
    bad.loc[0, "ref"] = "A" if table.loc[0, "ref"] != "A" else "C"
    enc = AlphaGenomeSiteEncoder(fasta=str(fasta), sequence_length=LENGTH,
                                 datamodule=VariantsModule(bad))
    with pytest.raises(ValueError, match="reference_mismatch"):
        enc.transform(None)


def test_states_dir_matches_the_model(setup, tmp_path):
    fasta, table, datamodule = setup
    variants = tmp_path / "variants.tsv"
    table.to_csv(variants, sep="\t", index=False)
    layers = ["dna_embedder", "transformer_tower"]
    assert cli.main(["--variants", str(variants), "--fasta", str(fasta), "--out",
                     str(tmp_path / "states"), "--sequence-length", str(LENGTH),
                     "--layers", *layers]) == 0
    live = AlphaGenomeSiteEncoder(layer="transformer_tower", fasta=str(fasta),
                                  sequence_length=LENGTH, datamodule=datamodule).transform(None)
    stored = AlphaGenomeSiteEncoder(layer="transformer_tower", states_dir=str(tmp_path / "states"),
                                    datamodule=datamodule).transform(None)
    np.testing.assert_array_equal(stored, live)
    features = AlphaGenomeSiteEncoder(output="features", feature_layers=layers,
                                      states_dir=str(tmp_path / "states"),
                                      datamodule=datamodule).transform(None)
    assert features.shape == (4, 6)
    # rows follow the datamodule, not the files
    reversed_module = VariantsModule(table.iloc[::-1].reset_index(drop=True))
    flipped = AlphaGenomeSiteEncoder(layer="transformer_tower", states_dir=str(tmp_path / "states"),
                                     datamodule=reversed_module).transform(None)
    np.testing.assert_array_equal(flipped, stored[::-1])
    with pytest.raises(KeyError):
        AlphaGenomeSiteEncoder(output="features", feature_layers=["up_res_block_0"],
                               states_dir=str(tmp_path / "states"),
                               datamodule=datamodule).transform(None)


def test_hydra_config_instantiates():
    from pathlib import Path

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    path = (Path(encoder_module.__file__).parents[1] / "configs" / "algorithms" / "latent"
            / "alphagenome_site.yaml")
    cfg = OmegaConf.load(path)
    cfg.states_dir = "unused"
    enc = instantiate(cfg)
    assert isinstance(enc, AlphaGenomeSiteEncoder)
    assert enc.layers == ["transformer_tower"] and enc.sequence_length == 1_048_576


def test_rows_follow_x(setup):
    fasta, _, datamodule = setup
    enc = AlphaGenomeSiteEncoder(layer="transformer_tower", fasta=str(fasta),
                                 sequence_length=LENGTH, datamodule=datamodule)
    every = enc.transform(None)
    x = np.array([[3.0], [1.0]], np.float32)
    np.testing.assert_allclose(enc.transform(x), every[[3, 1]], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(enc.transform(x[::-1]), every[[1, 3]], rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="row indices"):
        enc.transform(np.zeros((4, 3)))
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        enc.transform(np.array([[4.0]]))
    with pytest.raises(ValueError, match="whole-number"):
        enc.transform(np.array([[0.5]]))


def test_policy_reaches_load_local(monkeypatch, setup):
    fasta, _, datamodule = setup
    seen = {}
    model = make_model()

    def load_local(**kwargs):
        seen.update(kwargs)
        return model

    monkeypatch.setattr(model_module, "load_local", load_local)
    enc = AlphaGenomeSiteEncoder(fasta=str(fasta), sequence_length=LENGTH,
                                 datamodule=datamodule, policy="float32")
    enc.transform(None)
    assert seen["policy"] == "float32"
    with pytest.raises(ValueError, match="policy"):
        AlphaGenomeSiteEncoder(fasta=str(fasta), policy="fp8")
