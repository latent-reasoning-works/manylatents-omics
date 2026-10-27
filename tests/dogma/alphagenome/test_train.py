"""JaxTrainer on a toy regression with a frozen part."""
import json

import numpy as np
import pytest

pytest.importorskip("optax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from manylatents.dogma.alphagenome.train import (  # noqa: E402
    JaxTrainer, load_params, partition, save_params,
)


def _problem(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    frozen = {"projection": jnp.asarray(rng.normal(size=(5, 4)), jnp.float32)}
    true = rng.normal(size=(4, 2)).astype(np.float32)
    y = (x @ np.asarray(frozen["projection"]) @ true).astype(np.float32)
    trainable = {"head": {"w": jnp.zeros((4, 2)), "b": jnp.zeros((2,))}}

    def loss_fn(trainable, frozen, batch, rng):
        pred = batch["x"] @ frozen["projection"] @ trainable["head"]["w"] + trainable["head"]["b"]
        mse = jnp.mean((pred - batch["y"]) ** 2)
        return mse, {"mse": mse}

    return loss_fn, trainable, frozen, {"x": x, "y": y}


def test_loss_decreases_and_frozen_is_untouched():
    loss_fn, trainable, frozen, data = _problem()
    before = np.asarray(frozen["projection"]).copy()
    records = []
    trainer = JaxTrainer(lr=0.05, max_steps=200, batch_size=16, log_every=20,
                         callback=records.append)
    trained, history = trainer.fit(loss_fn, trainable, frozen, data)
    assert [r["step"] for r in history] == list(range(20, 201, 20))
    assert records == history
    assert history[-1]["loss"] < 1e-2 * history[0]["loss"]
    assert np.array_equal(np.asarray(frozen["projection"]), before)
    assert set(trained) == {"head"} and trained["head"]["w"].shape == (4, 2)
    assert {"loss", "mse", "grad_norm", "seconds"} <= set(history[0])


def test_batch_size_is_clipped_and_sgd_runs():
    loss_fn, trainable, frozen, data = _problem(n=6)
    trained, history = JaxTrainer(optimizer="sgd", lr=0.01, max_steps=3, batch_size=100,
                                  grad_clip=None, log_every=1).fit(
        loss_fn, trainable, frozen, data)
    assert len(history) == 3


def test_eval_early_stop_and_checkpoints(tmp_path):
    loss_fn, trainable, frozen, data = _problem()
    eval_data = {k: v[:10] for k, v in data.items()}
    trainer = JaxTrainer(lr=0.0, max_steps=100, batch_size=8, eval_every=5,
                         early_stop_patience=2, log_every=1000,
                         checkpoint_dir=str(tmp_path / "ckpt"))
    trained, history = trainer.fit(loss_fn, trainable, frozen, data, eval_data)
    # lr 0: the eval loss never improves after the first evaluation
    assert [r["step"] for r in history] == [5, 10, 15]
    assert "eval/loss" in history[0] and "eval/mse" in history[0]
    saved = load_params(tmp_path / "ckpt" / "trainable.npz")
    np.testing.assert_array_equal(saved["head"]["w"], np.asarray(trained["head"]["w"]))
    assert (tmp_path / "ckpt" / "trainable_best.npz").exists()
    assert json.loads((tmp_path / "ckpt" / "history.json").read_text()) == history


def test_save_and_load_round_trip(tmp_path):
    params = {"intervention": {"A": np.arange(6.0).reshape(3, 2), "b": np.ones(2)},
              "objective": {"track_head": {"layer_0": {"w": np.eye(2), "b": np.zeros(2)}}}}
    loaded = load_params(save_params(tmp_path / "p.npz", params))
    flat = jax.tree_util.tree_leaves_with_path
    assert [p for p, _ in flat(loaded)] == [p for p, _ in flat(params)]
    for (_, a), (_, b) in zip(flat(loaded), flat(params)):
        np.testing.assert_array_equal(a, b)


def test_partition():
    trainable, frozen = partition({"a": 1, "b": 2, "c": 3}, ["a", "c"])
    assert trainable == {"a": 1, "c": 3} and frozen == {"b": 2}
    with pytest.raises(KeyError):
        partition({"a": 1}, ["z"])


def test_non_finite_loss_raises():
    def loss_fn(trainable, frozen, batch, rng):
        return jnp.log(trainable["w"] - 1.0) * jnp.mean(batch["x"]), {}

    with pytest.raises(FloatingPointError):
        JaxTrainer(max_steps=2).fit(loss_fn, {"w": jnp.zeros(())}, {},
                                    {"x": np.ones((4, 1), np.float32)})


def test_bad_options():
    with pytest.raises(ValueError, match="optimizer"):
        JaxTrainer(optimizer="lamb")
    with pytest.raises(ValueError):
        JaxTrainer(max_steps=0)
