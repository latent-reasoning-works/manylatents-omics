"""Load AlphaGenome as a plain Haiku function, and build its inputs.

``alphagenome_research``'s ``AlphaGenomeModel`` hides the Haiku forward behind
methods that return track predictions only. Reading hidden states needs the
forward itself, so :func:`load_local` restores the same checkpoint into the same
Haiku transform and returns it with its parameters: nothing else.

Weights must already be in the local Hugging Face cache. Loading never opens a
login, never downloads and never calls the Atlas API.

The sequence helpers place the variant where ``alphagenome`` does for
``variant.reference_interval.resize(length)``: the window is centred on the
reference allele with the centre rounded up, so an SNV lands at index
``length // 2 - 1`` (524,287 at 1 Mb).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

# Input lengths the model was trained at, per alphagenome.models.dna_client.
# The Haiku model traces at other lengths, but those are extrapolations.
SUPPORTED_LENGTHS = frozenset({16_384, 131_072, 524_288, 1_048_576})
# alphagenome_research.model.dna_model.convert_to_organism_index
ORGANISM_INDEX = {"HOMO_SAPIENS": 0, "MUS_MUSCULUS": 1}
# alphagenome_research.model.dna_model.create_model
JMP_POLICY = "params=float32,compute=bfloat16,output=bfloat16"
# Named compute dtypes for ``policy``: the model's own bfloat16 policy, or
# float32 throughout (slower, more device memory, no bfloat16 rounding).
POLICIES = {"bfloat16": JMP_POLICY, "float32": "params=float32,compute=float32,output=float32"}


def policy_string(policy=None) -> str:
    """The jmp policy string for ``policy``: None or ``bfloat16`` (the default,
    AlphaGenome's own policy), ``float32``, or a jmp policy string."""
    if policy is None:
        return JMP_POLICY
    policy = str(policy)
    if policy in POLICIES:
        return POLICIES[policy]
    if "=" in policy:
        return policy
    raise ValueError(f"policy must be one of {sorted(POLICIES)} or a jmp policy string, "
                     f"got {policy!r}")
BASES = "ACGT"


@dataclass(frozen=True, eq=False)
class LocalModel:
    """A Haiku forward with its parameters.

    ``forward`` is an ``hk.TransformedWithState`` whose ``apply(params, state,
    rng, dna_sequence, organism_index)`` runs the model on one-hot input
    ``[B, L, 4]`` and organism indices ``[B]``. Equality is identity, so a model
    can key a cache of compiled functions.
    """

    params: Any
    state: Any
    forward: Any
    organism: str = "HOMO_SAPIENS"
    model_version: str = "fold_0"
    metadata: Any = None
    policy: Any = None
    supported_lengths: frozenset = SUPPORTED_LENGTHS

    @property
    def organism_index(self) -> int:
        return ORGANISM_INDEX[self.organism]

    def check_length(self, length: int) -> None:
        """Raise ValueError unless ``length`` is a trained input length."""
        if int(length) not in self.supported_lengths:
            raise ValueError(
                f"sequence length {length} is not one of AlphaGenome's trained lengths "
                f"{sorted(self.supported_lengths)}; pad the flanks with N at a supported "
                "length instead of shortening the input"
            )


def _organism_name(organism) -> str:
    name = getattr(organism, "name", organism)
    if name not in ORGANISM_INDEX:
        raise ValueError(f"organism must be one of {sorted(ORGANISM_INDEX)}, got {organism!r}")
    return name


def build_forward(metadata: Mapping, policy):
    """The Haiku transform ``alphagenome_research`` builds in ``create_model``."""
    import haiku as hk
    from alphagenome_research.model import model as ag_model

    @hk.transform_with_state
    def forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(ag_model.AlphaGenome, policy):
            return ag_model.AlphaGenome(metadata)(dna_sequence, organism_index)

    return forward


def load_local(model_version: str = "fold_0", organism: str = "HOMO_SAPIENS",
               hf_cache: Optional[str] = None, device=None,
               policy: Optional[str] = None) -> LocalModel:
    """Restore an AlphaGenome checkpoint from the local Hugging Face cache.

    Args:
        model_version: e.g. ``fold_0`` or ``all_folds``; the repository is
            ``google/alphagenome-<model_version with - for _>``.
        organism: ``HOMO_SAPIENS`` or ``MUS_MUSCULUS`` (or the ``alphagenome``
            enum); selects the organism index passed to the forward.
        hf_cache: Hugging Face cache directory; the default cache if None.
        device: JAX device for the parameters; the first local device if None.
            Unlike ``alphagenome_research``, a CPU device is accepted.
        policy: compute dtype, see :func:`policy_string`; the default is the
            bfloat16 policy that ``alphagenome_research`` uses.

    The checkpoint is restored as ``alphagenome_research.model.dna_model.create``
    does: output metadata for both organisms (the heads were trained with both),
    target shapes from tracing ``init``, and a strict Orbax restore.

    Raises:
        huggingface_hub.errors.LocalEntryNotFoundError: the weights are not cached.
    """
    import huggingface_hub
    import jax
    import jax.numpy as jnp
    import jmp
    import orbax.checkpoint as ocp
    from alphagenome.models import dna_model as ag_dna_model
    from alphagenome_research.model.metadata import metadata as metadata_lib

    organism = _organism_name(organism)
    checkpoint = huggingface_hub.snapshot_download(
        repo_id=f"google/alphagenome-{model_version.replace('_', '-').lower()}",
        cache_dir=hf_cache, local_files_only=True,
    )
    metadata = {o: metadata_lib.load(o) for o in (ag_dna_model.Organism.HOMO_SAPIENS,
                                                  ag_dna_model.Organism.MUS_MUSCULUS)}
    policy = jmp.get_policy(policy_string(policy))
    forward = build_forward(metadata, policy)
    target = jax.eval_shape(
        forward.init, jax.random.PRNGKey(0),
        jax.ShapeDtypeStruct((1, 2048, 4), jnp.float32),
        jax.ShapeDtypeStruct((1,), jnp.int32),
    )
    params, state = ocp.StandardCheckpointer().restore(checkpoint, target=target, strict=True)
    device = device or jax.local_devices()[0]
    return LocalModel(
        params=jax.device_put(params, device), state=jax.device_put(state, device),
        forward=forward, organism=organism, model_version=model_version,
        metadata=metadata, policy=policy,
    )


_ONE_HOT = np.zeros((256, 4), dtype=np.float32)
for _i, _base in enumerate(BASES):
    _ONE_HOT[ord(_base), _i] = 1
    _ONE_HOT[ord(_base.lower()), _i] = 1


def one_hot(sequence: str) -> np.ndarray:
    """``[len, 4]`` float32 in A, C, G, T order; any other letter (N) is all zeros.

    The same table as ``alphagenome_research``'s ``DNAOneHotEncoder``.
    """
    return _ONE_HOT[np.frombuffer(sequence.encode("latin1"), dtype=np.uint8)]


def site_window(pos: int, ref_length: int, length: int) -> tuple[int, int]:
    """``(start, site_index)``: 0-based window start and the variant's index in it.

    Matches ``genome.Variant(...).reference_interval.resize(length)``: the
    interval ``[pos - 1, pos - 1 + ref_length)`` keeps its centre, rounded up,
    and ``start = centre - (length + 1) // 2``.
    """
    if length <= 0:
        raise ValueError("length must be positive")
    start0 = int(pos) - 1
    centre = start0 + (int(ref_length) + 1) // 2
    start = centre - (int(length) + 1) // 2
    return start, start0 - start


def other_bases(ref: str, alt: str) -> tuple[str, str]:
    """The two bases that are neither ``ref`` nor ``alt``, in alphabetical order."""
    rest = [b for b in BASES if b not in (ref.upper(), alt.upper())]
    if len(rest) != 2:
        raise ValueError(f"need two distinct bases among ACGT, got {ref!r}>{alt!r}")
    return rest[0], rest[1]


def allele_bases(ref: str, alt: str, alleles: str = "all") -> dict[str, str]:
    """Allele name -> base: ``ref`` and ``alt``, plus ``alt2`` and ``alt3`` for ``all``."""
    if alleles not in ("all", "ref_alt"):
        raise ValueError(f"alleles must be 'all' or 'ref_alt', got {alleles!r}")
    out = {"ref": ref.upper(), "alt": alt.upper()}
    if alleles == "all":
        out["alt2"], out["alt3"] = other_bases(ref, alt)
    return out


def _fetch(fasta, chromosome: str, start: int, end: int) -> str:
    piece = fasta[chromosome][start:end]
    return str(getattr(piece, "seq", piece))


@dataclass(frozen=True)
class SiteSequences:
    """Input windows for one SNV: one sequence per allele, differing only at ``site_index``."""

    sequences: dict
    bases: dict
    site_index: int
    start: int
    reference_base: str

    @property
    def reference_matches(self) -> bool:
        return self.reference_base == self.bases["ref"]


def site_sequences(chromosome: str, pos: int, ref: str, alt: str, fasta, length: int,
                   alleles: str = "all") -> SiteSequences:
    """The reference window around an SNV, and one copy per allele.

    Args:
        chromosome: FASTA sequence name (``chr1``).
        pos: 1-based position.
        ref, alt: single bases.
        fasta: ``pyfaidx.Fasta``, or any mapping chromosome -> sliceable sequence.
        length: window length.
        alleles: ``all`` (ref, alt, alt2, alt3) or ``ref_alt``.

    The reference allele's window carries the FASTA base, whatever ``ref``
    says; compare ``reference_base`` with ``bases['ref']`` (``reference_matches``)
    to detect a mismatch.

    Raises:
        ValueError: not an SNV, or the window leaves the chromosome.
        KeyError: chromosome absent from the FASTA.
    """
    if len(ref) != 1 or len(alt) != 1:
        raise ValueError(f"not an SNV: {ref}>{alt}")
    bases = allele_bases(ref, alt, alleles)
    start, index = site_window(pos, 1, length)
    chromosome_length = len(fasta[chromosome])
    if start < 0 or start + length > chromosome_length:
        raise ValueError(f"window [{start}, {start + length}) leaves {chromosome} "
                         f"(length {chromosome_length})")
    window = _fetch(fasta, chromosome, start, start + length).upper()
    sequences = {name: window[:index] + base + window[index + 1:] for name, base in bases.items()}
    sequences["ref"] = window
    return SiteSequences(sequences=sequences, bases=bases, site_index=index, start=start,
                         reference_base=window[index])
