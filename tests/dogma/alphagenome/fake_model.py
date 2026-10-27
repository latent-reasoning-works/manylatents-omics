"""A tiny Haiku model with AlphaGenome's module names and resolution ladder.

The structure follows alphagenome_research.model.model (0.2.0): a sequence
encoder (DNA embedder, then six named down-resolution blocks each followed by
a 2x pool), the organism embedding added to the trunk, a nine-block
transformer tower at 128 bp (the real tower's residual loop: pair update every
second block, attention bias, ``x += mha(x)``, ``x += mlp(x)``), a seven-block
decoder back to 1 bp, and two output embedders (128 bp, then 1 bp). Widths are tiny; ``LENGTH`` stands in for a
trained input length.
"""
import haiku as hk
import jax
import jax.numpy as jnp
import jmp

from manylatents.dogma.alphagenome.model import LocalModel

LENGTH = 512


def _pool(x):
    b, s, d = x.shape
    return x.reshape(b, s // 2, 2, d).max(axis=2)


class DnaEmbedder(hk.Module):
    def __call__(self, dna_sequence, *, is_training):
        x = hk.Conv1D(output_channels=4, kernel_shape=5)(dna_sequence)
        return x + jax.nn.gelu(hk.Conv1D(output_channels=4, kernel_shape=3)(x))


class DownResBlock(hk.Module):
    def __call__(self, x, *, is_training):
        out = hk.Conv1D(output_channels=x.shape[-1] + 2, kernel_shape=3)(x)
        return out + jnp.pad(x, [(0, 0), (0, 0), (0, 2)])


class SequenceEncoder(hk.Module):
    def __call__(self, dna_sequence, *, is_training):
        intermediates = {}
        x = DnaEmbedder()(dna_sequence, is_training=is_training)
        intermediates["bin_size_1"] = x
        x = _pool(x)
        for block_idx, bin_size in enumerate([2, 4, 8, 16, 32, 64]):
            x = DownResBlock(f"downres_block_{block_idx}")(x, is_training=is_training)
            intermediates[f"bin_size_{bin_size}"] = x
            x = _pool(x)
        return x, intermediates


class PairUpdateBlock(hk.Module):
    """Reads the residual stream, as the real block does: here a [B, 1, 4] summary."""

    def __call__(self, x, pair_x, *, is_training):
        summary = jnp.mean(hk.Linear(4)(x), axis=1, keepdims=True)
        return summary if pair_x is None else pair_x + summary


class AttentionBiasBlock(hk.Module):
    def __call__(self, pair_x, *, is_training):
        return hk.Linear(1)(pair_x)


class MHABlock(hk.Module):
    def __call__(self, x, attention_bias, *, is_training):
        h = hk.Linear(x.shape[-1])(x)
        logits = h @ jnp.swapaxes(x, 1, 2) / x.shape[-1] + attention_bias
        weights = jax.nn.softmax(logits, axis=-1)
        return weights @ hk.Linear(x.shape[-1])(x)


class MLPBlock(hk.Module):
    def __call__(self, x, *, is_training):
        return hk.Linear(x.shape[-1])(jax.nn.relu(hk.Linear(2 * x.shape[-1])(x)))


class TransformerTower(hk.Module):
    """The loop of the real tower: the residual ``x`` feeds the pair update (every
    second block), the attention block and the MLP block, and each adds to ``x``."""

    def __call__(self, x, *, is_training):
        pair_x = None
        for i in range(9):
            if i % 2 == 0:
                pair_x = PairUpdateBlock()(x, pair_x, is_training=is_training)
            mha_bias = AttentionBiasBlock()(pair_x, is_training=is_training)
            x += MHABlock()(x, mha_bias, is_training=is_training)
            x += MLPBlock()(x, is_training=is_training)
        return x, pair_x


class UpResBlock(hk.Module):
    def __call__(self, x, unet_skip, *, is_training):
        channels = unet_skip.shape[-1]
        out = jnp.repeat(hk.Linear(channels)(x) + x[:, :, :channels], 2, axis=1)
        return out + hk.Linear(channels)(unet_skip)


class SequenceDecoder(hk.Module):
    def __call__(self, x, intermediates, *, is_training):
        for bin_size in [64, 32, 16, 8, 4, 2, 1]:
            x = UpResBlock()(x, intermediates[f"bin_size_{bin_size}"], is_training=is_training)
        return x


class OutputEmbedder(hk.Module):
    def __call__(self, x, organism_index, *, is_training, skip_x=None):
        x = hk.Linear(2 * x.shape[-1])(x)
        if skip_x is not None:
            skip_x = hk.Linear(x.shape[-1], with_bias=False)(skip_x)
            x += jnp.repeat(skip_x, x.shape[1] // skip_x.shape[1], axis=1)
        x += hk.Embed(2, x.shape[-1])(organism_index)[:, None, :]
        return jax.nn.gelu(x)


class AlphaGenome(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name or "alphagenome")

    def __call__(self, dna_sequence, organism_index, *, is_training=False):
        trunk, intermediates = SequenceEncoder()(dna_sequence, is_training=is_training)
        trunk += hk.Embed(2, trunk.shape[-1])(organism_index)[:, None, :]
        trunk, pair = TransformerTower()(trunk, is_training=is_training)
        x = SequenceDecoder()(trunk, intermediates, is_training=is_training)
        embeddings_128bp = OutputEmbedder()(trunk, organism_index, is_training=is_training)
        embeddings_1bp = OutputEmbedder()(x, organism_index, is_training=is_training,
                                          skip_x=embeddings_128bp)
        predictions = {"embeddings_1bp": embeddings_1bp,
                       "track": hk.Linear(3)(embeddings_128bp)}
        return predictions, (embeddings_1bp, embeddings_128bp)


def make_model(policy="params=float32,compute=bfloat16,output=bfloat16", seed=0,
               organism="HOMO_SAPIENS") -> LocalModel:
    policy = jmp.get_policy(policy)

    @hk.transform_with_state
    def forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(AlphaGenome, policy):
            return AlphaGenome()(dna_sequence, organism_index)

    params, state = forward.init(jax.random.PRNGKey(seed), jnp.zeros((1, LENGTH, 4)),
                                 jnp.zeros((1,), jnp.int32))
    return LocalModel(params=params, state=state, forward=forward, organism=organism,
                      model_version="fake", policy=policy,
                      supported_lengths=frozenset({LENGTH}))
