"""A small optax training loop for parameters that sit beside a frozen model.

:class:`JaxTrainer` optimises a ``trainable`` pytree (intervention and head
parameters) while a ``frozen`` pytree (all AlphaGenome parameters and state)
is only read: gradients are taken with respect to ``trainable`` alone, and
``frozen`` is passed through the jitted step unchanged. manylatents' Lightning
trainer never sees this loop; the trainer is a nested Hydra object inside the
algorithm's config. No torch here.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

OPTIMIZERS = ("adamw", "adam", "sgd")


def partition(params: dict, trainable_keys) -> tuple[dict, dict]:
    """Split a dict pytree at its top level: ``(trainable, frozen)``."""
    keys = set(trainable_keys)
    unknown = keys - set(params)
    if unknown:
        raise KeyError(f"trainable keys not in params: {sorted(unknown)}")
    return ({k: v for k, v in params.items() if k in keys},
            {k: v for k, v in params.items() if k not in keys})


def _flatten(tree) -> dict:
    import jax

    flat = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        key = "/".join(str(getattr(p, "key", getattr(p, "idx", p))) for p in path)
        flat[key] = np.asarray(leaf)
    return flat


def save_params(path, params) -> Path:
    """Write a pytree of arrays (nested dicts) to ``.npz``, keys joined by ``/``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **_flatten(params))
    return path


def load_params(path) -> dict:
    """The nested dict written by :func:`save_params`; empty dicts are not kept."""
    out: dict = {}
    with np.load(path) as data:
        for key in data.files:
            node = out
            *parents, leaf = key.split("/")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = data[key]
    return out


class JaxTrainer:
    """Optimise ``trainable`` against ``loss_fn`` with optax.

    Args:
        optimizer: ``adamw`` (default), ``adam`` or ``sgd``.
        lr, weight_decay: learning rate; weight decay for adamw.
        max_steps: optimiser steps.
        batch_size: examples per step; minibatches are drawn by shuffling each
            epoch and dropping the last partial batch (one compiled shape).
            Clipped to the number of examples.
        grad_clip: global-norm clip; 0 or None disables it.
        seed: shuffling and step PRNG keys.
        log_every: log the step's metrics every N steps (and at the last).
        eval_every: evaluate on ``eval_data`` every N steps (0: never).
        early_stop_patience: stop after this many evaluations without a lower
            eval loss (0: never). Without ``eval_data`` the training loss of
            the evaluation step is used.
        checkpoint_dir: if set, the trainable parameters go to
            ``trainable.npz`` there at the end (and ``trainable_best.npz`` at
            each improvement when evaluating), with ``history.json``.
        remat: rematerialise the model's blocks in the backward pass. Read by
            the forward builder (:func:`~.interventions.make_forward`); trades
            compute for activation memory on long inputs.
        callback: called as ``callback(record)`` with every logged record, e.g.
            to forward metrics to wandb.

    ``loss_fn(trainable, frozen, batch, rng) -> (loss, metrics)``, where
    ``metrics`` is a dict of scalars. AlphaGenome's own jmp policy (bf16
    compute on GPU) applies inside its forward; trainable parameters and the
    optimiser state stay float32.
    """

    def __init__(self, optimizer: str = "adamw", lr: float = 1e-3, weight_decay: float = 0.0,
                 max_steps: int = 1000, batch_size: int = 16, grad_clip: Optional[float] = 1.0,
                 seed: int = 0, log_every: int = 50, eval_every: int = 0,
                 early_stop_patience: int = 0, checkpoint_dir: Optional[str] = None,
                 remat: bool = False, callback: Optional[Callable] = None):
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {optimizer!r}")
        if int(max_steps) < 1 or int(batch_size) < 1:
            raise ValueError("max_steps and batch_size must be >= 1")
        self.optimizer = optimizer
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_steps = int(max_steps)
        self.batch_size = int(batch_size)
        self.grad_clip = float(grad_clip) if grad_clip else 0.0
        self.seed = int(seed)
        self.log_every = max(1, int(log_every))
        self.eval_every = int(eval_every)
        self.early_stop_patience = int(early_stop_patience)
        self.checkpoint_dir = checkpoint_dir
        self.remat = bool(remat)
        self.callback = callback

    def make_optimizer(self):
        import optax

        if self.optimizer == "adamw":
            opt = optax.adamw(self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "adam":
            opt = optax.adam(self.lr)
        else:
            opt = optax.sgd(self.lr)
        return optax.chain(optax.clip_by_global_norm(self.grad_clip), opt) if self.grad_clip \
            else opt

    def _emit(self, record: dict) -> None:
        logger.info(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in record.items()))
        if self.callback is not None:
            self.callback(record)

    @staticmethod
    def _n(data: dict) -> int:
        sizes = {len(v) for v in data.values()}
        if len(sizes) != 1:
            raise ValueError(f"all data arrays need the same first dimension, got {sizes}")
        return sizes.pop()

    def evaluate(self, eval_step, trainable, frozen, data: dict, rng) -> dict:
        """Mean loss and metrics over full batches of ``data`` (the remainder is
        a last, padded batch weighted by its real size)."""
        n = self._n(data)
        bs = min(self.batch_size, n)
        totals: dict = {}
        for begin in range(0, n, bs):
            index = np.arange(begin, begin + bs)
            real = int(np.sum(index < n))
            index = np.minimum(index, n - 1)
            batch = {k: v[index] for k, v in data.items()}
            loss, metrics = eval_step(trainable, frozen, batch, rng)
            for k, v in {"loss": loss, **metrics}.items():
                totals[k] = totals.get(k, 0.0) + float(v) * real
        return {k: v / n for k, v in totals.items()}

    def fit(self, loss_fn: Callable, trainable, frozen, data: dict,
            eval_data: Optional[dict] = None):
        """Train; returns ``(trainable, history)``.

        ``data`` and ``eval_data`` map names to arrays with a shared first
        dimension (examples); each step slices a minibatch of every array.
        ``history`` is a list of dicts: ``step``, ``loss``, the metrics,
        ``seconds``, and ``eval/...`` entries at evaluation steps.
        """
        import jax
        import jax.numpy as jnp
        import optax

        n = self._n(data)
        bs = min(self.batch_size, n)
        opt = self.make_optimizer()
        trainable = jax.tree_util.tree_map(jnp.asarray, trainable)
        opt_state = opt.init(trainable)

        # optax >= 0.2.5 moved global_norm to optax.tree.norm
        norm = getattr(getattr(optax, "tree", None), "norm", None) or optax.global_norm

        @jax.jit
        def step(trainable, opt_state, frozen, batch, rng):
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                trainable, frozen, batch, rng)
            updates, opt_state = opt.update(grads, opt_state, trainable)
            return optax.apply_updates(trainable, updates), opt_state, loss, metrics, \
                norm(grads)

        eval_step = jax.jit(loss_fn)
        shuffle = np.random.default_rng(self.seed)
        key = jax.random.PRNGKey(self.seed)
        history: list[dict] = []
        order, cursor = shuffle.permutation(n), 0
        best, best_step, stale = np.inf, -1, 0
        tick = time.time()
        for i in range(1, self.max_steps + 1):
            if cursor + bs > n:
                order, cursor = shuffle.permutation(n), 0
            index = order[cursor:cursor + bs]
            cursor += bs
            batch = {k: v[index] for k, v in data.items()}
            key, sub = jax.random.split(key)
            trainable, opt_state, loss, metrics, gnorm = step(trainable, opt_state, frozen,
                                                              batch, sub)
            if not np.isfinite(float(loss)):
                raise FloatingPointError(f"loss is {float(loss)} at step {i}")
            record = None
            if i % self.log_every == 0 or i == self.max_steps:
                record = {"step": i, "loss": float(loss), "grad_norm": float(gnorm),
                          **{k: float(v) for k, v in metrics.items()},
                          "seconds": time.time() - tick}
            if self.eval_every and i % self.eval_every == 0:
                record = record or {"step": i, "loss": float(loss),
                                    "seconds": time.time() - tick}
                evaluated = (self.evaluate(eval_step, trainable, frozen, eval_data, sub)
                             if eval_data is not None else {"loss": float(loss)})
                record.update({f"eval/{k}": v for k, v in evaluated.items()})
                if evaluated["loss"] < best:
                    best, best_step, stale = evaluated["loss"], i, 0
                    if self.checkpoint_dir:
                        save_params(Path(self.checkpoint_dir) / "trainable_best.npz", trainable)
                else:
                    stale += 1
            if record is not None:
                history.append(record)
                self._emit(record)
            if self.early_stop_patience and stale >= self.early_stop_patience:
                logger.info("early stop at step %d (best eval loss %.5g at step %d)",
                            i, best, best_step)
                break
        if self.checkpoint_dir:
            save_params(Path(self.checkpoint_dir) / "trainable.npz", trainable)
            (Path(self.checkpoint_dir) / "history.json").write_text(json.dumps(history, indent=1))
        return trainable, history

    def __repr__(self):
        return (f"JaxTrainer(optimizer={self.optimizer!r}, lr={self.lr}, "
                f"max_steps={self.max_steps}, batch_size={self.batch_size})")
