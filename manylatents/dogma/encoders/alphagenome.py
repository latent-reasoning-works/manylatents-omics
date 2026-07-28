"""AlphaGenome encoder for DNA sequences with regulatory predictions.

AlphaGenome is a JAX-based foundation model for genomics that predicts
regulatory features at single base-pair resolution across 1Mb context.

References:
    - Paper: "AlphaGenome: Foundation model for the human genome"
    - GitHub: https://github.com/google-deepmind/alphagenome_research
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from manylatents.algorithms.latent.foundation_encoder import FoundationEncoder


class AlphaGenomeEncoder(FoundationEncoder):
    """AlphaGenome encoder for DNA sequences.

    Provides both embeddings (encode) and regulatory track predictions (predict).
    Uses JAX internally with transparent PyTorch tensor conversion.

    Args:
        model_name: Model variant. Options: "alphagenome" (1bp), "alphagenome_128bp".
        layer_name: Embedding resolution. Options: "embeddings_1bp", "embeddings_128bp".
        weights_path: Local path to weights. If None, downloads from HuggingFace.
        device: Device for PyTorch output tensors ("cuda" or "cpu").

    Example:
        >>> encoder = AlphaGenomeEncoder()
        >>> embedding = encoder.encode("ATGAAGTTTGGCGTCCGTGCCTGA")
        >>> predictions = encoder.predict("ATGAAGTTTGGCGTCCGTGCCTGA")
    """

    # Model configurations from architecture triage
    MODELS = {
        "alphagenome": {
            "default_layer": "embeddings_1bp",
            "embedding_dim": 1536,
            "context_length": 1_000_000,
        },
        "alphagenome_128bp": {
            "default_layer": "embeddings_128bp",
            "embedding_dim": 3072,
            "context_length": 1_000_000,
        },
    }

    DEFAULT_WEIGHTS = "/network/weights/alphagenome/"

    # Input lengths the model is trained for, per alphagenome.models.dna_client.
    # The Haiku model will happily *trace* at other lengths (2 kb included), but
    # those are outside the training distribution — see encode_layers().
    SUPPORTED_LENGTHS = (16_384, 131_072, 524_288, 1_048_576)

    def __init__(
        self,
        model_name: str = "alphagenome",
        layer_name: Optional[str] = None,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        if model_name not in self.MODELS:
            raise ValueError(
                f"model_name must be one of {list(self.MODELS.keys())}, got {model_name}"
            )

        self.model_name = model_name
        self.layer_name = layer_name or self.MODELS[model_name]["default_layer"]
        self.weights_path = weights_path
        self._embedding_dim = self.MODELS[model_name]["embedding_dim"]
        self._context_length = self.MODELS[model_name]["context_length"]

        self._model = None
        self._jax_to_torch = None
        self._torch_to_jax = None

    def _load_model(self) -> None:
        """Lazy load AlphaGenome JAX model."""
        if self._model is not None:
            return

        try:
            from alphagenome_research.model import dna_model
            from torch_jax_interop import jax_to_torch, torch_to_jax
        except ImportError as e:
            raise ImportError(
                "AlphaGenome requires 'alphagenome-research' and 'torch-jax-interop'. "
                "Install with: uv sync --extra alphagenome"
            ) from e

        # Patch for JAX < 0.5 compatibility (jax.memory.Space.Host not available)
        self._patch_jax_memory_compat(dna_model)

        self._jax_to_torch = jax_to_torch
        self._torch_to_jax = torch_to_jax
        self._dna_model = dna_model

        # Try local weights first, fall back to HuggingFace
        if self.weights_path and Path(self.weights_path).exists():
            self._model = dna_model.create_from_huggingface(self.weights_path)
        else:
            self._model = dna_model.create_from_huggingface("all_folds")

    @staticmethod
    def _patch_jax_memory_compat(dna_model_module) -> None:
        """Patch alphagenome_research for JAX < 0.5 compatibility.

        The alphagenome_research library uses jax.memory.Space.Host which was
        added in newer JAX versions. This patches the module to use jax.device_get
        instead, which provides equivalent functionality.
        """
        import functools
        import jax

        # Check if jax.memory exists
        if hasattr(jax, "memory"):
            return  # No patch needed

        # Monkey-patch the _upcast_single_batch_predictions function
        original_func = dna_model_module._upcast_single_batch_predictions

        @functools.partial(jax.jit, static_argnames=["transfer_to_host"])
        def patched_upcast_single_batch_predictions(x, *, transfer_to_host=True):
            """Patched version that uses jax.device_get instead of jax.memory.Space."""
            from alphagenome import tensor_utils

            x = jax.tree.map(lambda arr: tensor_utils.upcast_floating(arr[0]), x)
            return jax.device_get(x) if transfer_to_host else x

        dna_model_module._upcast_single_batch_predictions = (
            patched_upcast_single_batch_predictions
        )

    def encode(
        self,
        sequence: str,
        chunk_size: Optional[int] = None,
        overlap: int = 0,
        aggregate: Optional[str] = "mean",
    ) -> Tensor:
        """Encode DNA sequence into embedding space.

        Args:
            sequence: DNA nucleotide sequence (ACGT alphabet).
            chunk_size: Process in chunks of this size. If None, process whole sequence.
            overlap: Overlap between chunks in base pairs.
            aggregate: How to combine chunks ("mean", "max", or None for raw chunks).

        Returns:
            Embedding tensor of shape (1, embedding_dim) if aggregated,
            or (num_chunks, embedding_dim) if aggregate=None.
        """
        self._ensure_loaded()

        # Short sequence - no chunking needed
        if not chunk_size or len(sequence) <= chunk_size:
            jax_emb = self._forward_embedding(sequence)
            return self._jax_to_torch(jax_emb).to(self.device)

        # Chunked encoding
        chunk_embeddings = self._encode_chunked(sequence, chunk_size, overlap)

        if aggregate is None:
            return chunk_embeddings
        elif aggregate == "mean":
            return chunk_embeddings.mean(dim=0, keepdim=True)
        elif aggregate == "max":
            return chunk_embeddings.max(dim=0, keepdim=True).values
        else:
            raise ValueError(f"aggregate must be 'mean', 'max', or None, got {aggregate}")

    def predict(
        self,
        sequence: str,
        chunk_size: Optional[int] = None,
        overlap: int = 0,
        aggregate: Optional[str] = None,
        output_types: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """Predict regulatory tracks for DNA sequence.

        Args:
            sequence: DNA nucleotide sequence (ACGT alphabet).
            chunk_size: Process in chunks of this size. If None, process whole sequence.
            overlap: Overlap between chunks in base pairs.
            aggregate: How to combine chunks (None recommended for position-specific tracks).
            output_types: List of output types to request. If None, returns all available.
                Options: "ATAC", "CAGE", "CHIP_HISTONE", "CHIP_TF", "DNASE", "RNA_SEQ", etc.

        Returns:
            Dict mapping track names to PyTorch tensors.
        """
        self._ensure_loaded()

        # Short sequence - no chunking needed
        if not chunk_size or len(sequence) <= chunk_size:
            preds = self._forward_predict(sequence, output_types)
            return {
                name: self._to_torch(arr).to(self.device)
                for name, arr in preds.items()
            }

        # Chunked prediction
        return self._predict_chunked(sequence, chunk_size, overlap, aggregate, output_types)

    def _forward_embedding(self, sequence: str) -> Any:
        """Run forward pass and extract embedding from specified layer.

        Args:
            sequence: DNA sequence.

        Returns:
            JAX array of embeddings, mean-pooled over sequence length.
            Shape: (1, embedding_dim)
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        # Get raw model output by calling internal apply_fn
        # AlphaGenome's public API filters out embeddings, so we need to call
        # the underlying Haiku model directly
        predictions = self._get_raw_predictions(sequence)

        # Extract embeddings based on layer_name
        if self.layer_name == "embeddings_1bp":
            embeddings = predictions.get("embeddings_1bp")
            if embeddings is None:
                raise KeyError(
                    f"embeddings_1bp not found in predictions. "
                    f"Available keys: {list(predictions.keys())}"
                )
            # Mean pool over sequence length: (1, S, 1536) -> (1, 1536)
            return jnp.mean(embeddings, axis=1)

        elif self.layer_name == "embeddings_128bp":
            # For 128bp resolution, we'd need to access trunk embeddings
            # This requires different extraction logic
            raise NotImplementedError(
                "embeddings_128bp extraction not yet implemented. "
                "Use layer_name='embeddings_1bp' for now."
            )

        else:
            raise ValueError(
                f"Unknown layer_name: {self.layer_name}. "
                f"Supported: 'embeddings_1bp', 'embeddings_128bp'"
            )

    def _build_transformed(self) -> Any:
        """Rebuild AlphaGenome as a Haiku transform.

        The public API filters embeddings out of its predictions, so both the
        embedding path and the layer-tap path need their own forward.
        """
        if getattr(self, "_transformed", None) is not None:
            return self._transformed

        import haiku as hk
        import jmp
        from alphagenome.models import dna_model as dna_model_lib
        from alphagenome_research.model import model as ag_model
        from alphagenome_research.model.metadata import metadata as metadata_lib

        # Metadata for BOTH organisms (the model was trained with both)
        metadata = {
            dna_model_lib.Organism.HOMO_SAPIENS: metadata_lib.load(
                dna_model_lib.Organism.HOMO_SAPIENS
            ),
            dna_model_lib.Organism.MUS_MUSCULUS: metadata_lib.load(
                dna_model_lib.Organism.MUS_MUSCULUS
            ),
        }

        jmp_policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")

        @hk.transform_with_state
        def _forward(dna_sequence, organism_index):
            with hk.mixed_precision.push_policy(ag_model.AlphaGenome, jmp_policy):
                return ag_model.AlphaGenome(metadata)(dna_sequence, organism_index)

        self._transformed = _forward
        return _forward

    def _encode_one_hot(self, sequence: str) -> Any:
        """One-hot encode and add a batch dim: (1, S, 4)."""
        import jax.numpy as jnp

        one_hot = self._model._one_hot_encoder.encode(sequence)
        return jnp.asarray(one_hot)[jnp.newaxis]

    @staticmethod
    def _organism_index() -> Any:
        """Organism selector; human is index 0."""
        import jax.numpy as jnp

        return jnp.array([0], dtype=jnp.int32)

    def _get_raw_predictions(self, sequence: str) -> Any:
        """Get raw model predictions including embeddings.

        The public AlphaGenome API filters out embeddings via extract_predictions().
        This method creates a custom forward pass to access embeddings directly.

        Args:
            sequence: DNA sequence.

        Returns:
            Dict of raw predictions including embeddings_1bp.
        """
        import jax

        if not hasattr(self, "_raw_predict_fn"):
            forward = self._build_transformed()

            def _apply_fn(params, state, dna_sequence, organism_index):
                (predictions, _), _ = forward.apply(
                    params, state, None, dna_sequence, organism_index
                )
                return predictions

            self._raw_predict_fn = jax.jit(_apply_fn)

        return self._raw_predict_fn(
            self._model._params,
            self._model._state,
            self._encode_one_hot(sequence),
            self._organism_index(),
        )

    def available_layers(self, sequence_length: int = 16_384) -> Dict[str, List[int]]:
        """Enumerate tappable internal modules and their output shapes.

        AlphaGenome is JAX/Haiku, so there is no torch forward hook. The
        equivalent is ``hk.intercept_methods``, which observes every module
        ``__call__``. This traces the model without spending FLOPs and reports
        what can be tapped — notably the nine transformer blocks at
        ``alphagenome/transformer_tower/mha_block[_N]``.

        Args:
            sequence_length: Length to trace at. Shapes downstream of the
                sequence encoder scale with it.

        Returns:
            Mapping of module path to output shape.
        """
        self._ensure_loaded()

        import haiku as hk
        import jax

        forward = self._build_transformed()
        seen: Dict[str, List[int]] = {}

        def interceptor(next_f, args, kwargs, context):
            out = next_f(*args, **kwargs)
            if context.method_name == "__call__":
                name = getattr(context.module, "module_name", None)
                arr = out[0] if isinstance(out, (tuple, list)) and out else out
                shape = getattr(arr, "shape", None)
                if name is not None and shape is not None:
                    seen[name] = list(shape)
            return out

        def _traced(params, state, dna_sequence, organism_index):
            with hk.intercept_methods(interceptor):
                (predictions, _), _ = forward.apply(
                    params, state, None, dna_sequence, organism_index
                )
            return predictions

        jax.eval_shape(
            _traced,
            self._model._params,
            self._model._state,
            self._encode_one_hot("A" * sequence_length),
            self._organism_index(),
        )
        return seen

    def encode_layers(
        self,
        sequence: str,
        layers: List[str],
        aggregate: Optional[str] = "mean",
    ) -> Dict[str, Tensor]:
        """Extract activations from named internal modules.

        The per-layer counterpart to `encode()`, which returns only the final
        embedding. Discover valid names with `available_layers()`.

        Args:
            sequence: DNA sequence. Its length should be one of
                SUPPORTED_LENGTHS; the model traces at other lengths but was
                not trained there, so shorter inputs are a silent extrapolation.
            layers: Module paths to tap, e.g.
                ``["alphagenome/transformer_tower/mha_block_0"]``.
            aggregate: Pool over the positional axes ("mean", "max", or None to
                keep the full activation).

        Returns:
            Mapping of module path to a PyTorch tensor.
        """
        if not layers:
            raise ValueError("layers must name at least one module to tap")

        if len(sequence) not in self.SUPPORTED_LENGTHS:
            warnings.warn(
                f"sequence length {len(sequence)} is not one of AlphaGenome's "
                f"trained lengths {self.SUPPORTED_LENGTHS}. The model will still "
                f"run, but the result is an extrapolation — to match a smaller "
                f"receptive field, pad the flanks with 'N' at a supported length "
                f"rather than shortening the input.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._ensure_loaded()

        import haiku as hk
        import jax
        import jax.numpy as jnp

        wanted = tuple(layers)
        if not hasattr(self, "_tap_fns"):
            self._tap_fns: Dict[tuple, Any] = {}

        if wanted not in self._tap_fns:
            forward = self._build_transformed()
            keep = set(wanted)

            def _apply_fn(params, state, dna_sequence, organism_index):
                caught: Dict[str, Any] = {}

                def interceptor(next_f, args, kwargs, context):
                    out = next_f(*args, **kwargs)
                    if context.method_name == "__call__":
                        name = getattr(context.module, "module_name", None)
                        if name in keep:
                            caught[name] = (
                                out[0]
                                if isinstance(out, (tuple, list)) and out
                                else out
                            )
                    return out

                with hk.intercept_methods(interceptor):
                    forward.apply(params, state, None, dna_sequence, organism_index)
                return caught

            self._tap_fns[wanted] = jax.jit(_apply_fn)

        caught = self._tap_fns[wanted](
            self._model._params,
            self._model._state,
            self._encode_one_hot(sequence),
            self._organism_index(),
        )

        missing = [name for name in wanted if name not in caught]
        if missing:
            raise KeyError(
                f"modules not reached during the forward pass: {missing}. "
                f"Call available_layers() for valid names."
            )

        out: Dict[str, Tensor] = {}
        for name, arr in caught.items():
            # Activations are (1, L, D); pair-track modules are (1, P, P, F).
            # Pool everything between batch and channel so both shapes reduce
            # to (1, D).
            if aggregate is not None and arr.ndim > 2:
                axes = tuple(range(1, arr.ndim - 1))
                if aggregate == "mean":
                    arr = jnp.mean(arr, axis=axes)
                elif aggregate == "max":
                    arr = jnp.max(arr, axis=axes)
                else:
                    raise ValueError(
                        f"aggregate must be 'mean', 'max', or None, got {aggregate}"
                    )
            out[name] = self._jax_to_torch(
                arr.astype(jnp.float32)
            ).to(self.device)
        return out

    def _forward_predict(
        self, sequence: str, output_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run forward pass and return all prediction tracks.

        Args:
            sequence: DNA sequence.
            output_types: Optional list of output type names to request.
                Options: "ATAC", "CAGE", "CHIP_HISTONE", "CHIP_TF",
                         "DNASE", "RNA_SEQ", "SPLICE_SITES", etc.

        Returns:
            Dict of track name (lowercase) to JAX array.
        """
        OutputType = self._dna_model.OutputType

        if output_types is None:
            # Default to commonly used output types
            requested = [
                OutputType.ATAC,
                OutputType.CAGE,
                OutputType.DNASE,
                OutputType.RNA_SEQ,
            ]
        else:
            requested = [OutputType[ot.upper()] for ot in output_types]

        output = self._model.predict_sequence(
            sequence,
            requested_outputs=requested,
            ontology_terms=None,
        )

        results = {}
        for ot in requested:
            track_data = output.get(ot)
            if track_data is not None and track_data.values is not None:
                results[ot.name.lower()] = track_data.values

        return results

    def _encode_chunked(
        self,
        sequence: str,
        chunk_size: int,
        overlap: int,
    ) -> Tensor:
        """Encode long sequence in overlapping chunks."""
        chunks = self._split_sequence(sequence, chunk_size, overlap)
        embeddings = [self._forward_embedding(chunk) for chunk in chunks]

        import jax.numpy as jnp

        stacked = jnp.stack(embeddings, axis=0)
        return self._jax_to_torch(stacked).to(self.device)

    def _predict_chunked(
        self,
        sequence: str,
        chunk_size: int,
        overlap: int,
        aggregate: Optional[str],
        output_types: Optional[List[str]],
    ) -> Dict[str, Tensor]:
        """Predict on long sequence in overlapping chunks."""
        chunks = self._split_sequence(sequence, chunk_size, overlap)
        chunk_preds = [self._forward_predict(chunk, output_types) for chunk in chunks]

        if not chunk_preds:
            return {}

        track_names = chunk_preds[0].keys()
        import jax.numpy as jnp

        result = {}
        for name in track_names:
            stacked = jnp.stack([p[name] for p in chunk_preds], axis=0)
            tensor = self._jax_to_torch(stacked).to(self.device)

            if aggregate == "mean":
                tensor = tensor.mean(dim=0, keepdim=True)
            elif aggregate == "max":
                tensor = tensor.max(dim=0, keepdim=True).values
            # else: keep as (num_chunks, ...)

            result[name] = tensor

        return result

    @staticmethod
    def _split_sequence(sequence: str, chunk_size: int, overlap: int) -> List[str]:
        """Split sequence into overlapping chunks."""
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

        step = chunk_size - overlap
        chunks = []
        for i in range(0, len(sequence), step):
            chunk = sequence[i : i + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)
        return chunks

    def _to_torch(self, arr: Any) -> Tensor:
        """Convert array (JAX or numpy) to PyTorch tensor."""
        import numpy as np

        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr)
        else:
            # JAX array
            return self._jax_to_torch(arr)

    @property
    def modality(self) -> str:
        return "dna"

    @property
    def context_length(self) -> int:
        """Maximum sequence length the model can process."""
        return self._context_length
