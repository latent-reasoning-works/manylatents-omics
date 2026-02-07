"""Native Orthrus RNA encoder using mamba-ssm 2.x.

Re-implementation of Orthrus inference that works with mamba-ssm>=2.0,
avoiding the version conflict with Evo2.

This module implements the minimal MixerModel architecture from Orthrus
and loads pretrained weights from HuggingFace.

References:
    - Paper: Fradkin et al. (2024) "Orthrus: Towards Evolutionary and Functional RNA Foundation Models"
    - Original: https://github.com/bowang-lab/Orthrus
    - HuggingFace: https://huggingface.co/quietflamingo/orthrus-base-4-track
"""

from __future__ import annotations

import math
from functools import partial
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from manylatents.dogma.encoders.base import FoundationEncoder


# Nucleotide vocabulary for one-hot encoding
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 0}


def _one_hot_encode(sequence: str, device: str = "cpu") -> Tensor:
    """One-hot encode an RNA sequence.

    Args:
        sequence: RNA sequence (A, C, G, U). T is converted to U.
        device: Target device.

    Returns:
        Tensor of shape (1, 4, L) - batch, channels, length
    """
    sequence = sequence.upper().replace("T", "U")
    indices = [NUC_TO_IDX.get(n, 0) for n in sequence]

    one_hot = torch.zeros(1, len(sequence), 4, device=device)
    for i, idx in enumerate(indices):
        one_hot[0, i, idx] = 1.0

    # Transpose to (B, C, L) format expected by MixerModel
    return one_hot.transpose(1, 2)


def _mean_unpadded(x: Tensor, lengths: Tensor) -> Tensor:
    """Mean pool over sequence length, respecting padding.

    Args:
        x: Hidden states of shape (B, L, H)
        lengths: Actual sequence lengths of shape (B,)

    Returns:
        Mean-pooled tensor of shape (B, H)
    """
    mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
    masked = x * mask.unsqueeze(-1)
    return masked.sum(dim=1) / lengths.unsqueeze(-1).float()


class MixerModel(nn.Module):
    """Mamba-based sequence model for RNA.

    Re-implementation of Orthrus MixerModel compatible with mamba-ssm 2.x.
    """

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        input_dim: int = 4,  # One-hot RNA (A, C, G, U)
        ssm_cfg: Optional[dict] = None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.embedding = nn.Linear(input_dim, d_model, **factory_kwargs)

        # Import mamba-ssm components (2.x API)
        from mamba_ssm.modules.mamba_simple import Mamba
        from mamba_ssm.modules.block import Block
        try:
            from mamba_ssm.ops.triton.layer_norm import RMSNorm
        except ImportError:
            # Fallback for different mamba-ssm versions
            RMSNorm = None

        if ssm_cfg is None:
            ssm_cfg = {}

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            mixer_cls = partial(Mamba, layer_idx=i, **ssm_cfg, **factory_kwargs)

            if rms_norm and RMSNorm is not None:
                norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)
            else:
                norm_cls = partial(nn.LayerNorm, eps=norm_epsilon, **factory_kwargs)

            # mamba-ssm 2.x requires mlp_cls argument
            block = Block(
                d_model,
                mixer_cls,
                mlp_cls=nn.Identity,  # No MLP in Orthrus architecture
                norm_cls=norm_cls,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
            )
            block.layer_idx = i
            self.layers.append(block)

        self.norm_f = nn.LayerNorm(d_model, eps=norm_epsilon, **factory_kwargs)

        # Initialize weights
        self.apply(partial(self._init_weights, n_layer=n_layer))

    def _init_weights(self, module: nn.Module, n_layer: int):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, x: Tensor, channel_last: bool = False,
        capture_layers: Optional[Set[int]] = None,
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        """Forward pass with optional intermediate capture.

        Args:
            x: Input tensor of shape (B, C, L) or (B, L, C) if channel_last
            channel_last: If True, input is (B, L, C)
            capture_layers: Set of layer indices to capture intermediates from.

        Returns:
            Tuple of (final hidden states, dict of captured intermediates).
            Intermediates dict is empty if capture_layers is None.
        """
        if not channel_last:
            x = x.transpose(1, 2)

        hidden_states = self.embedding(x)
        residual = None
        intermediates: Dict[int, Tensor] = {}

        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual)
            if capture_layers and i in capture_layers:
                combined = (hidden_states + residual) if residual is not None else hidden_states
                intermediates[i] = self.norm_f(combined.to(dtype=self.norm_f.weight.dtype))

        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        return hidden_states, intermediates

    def representation(
        self, x: Tensor, lengths: Tensor, channel_last: bool = False,
        capture_layers: Optional[Set[int]] = None,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Get mean-pooled representation.

        Args:
            x: Input tensor
            lengths: Sequence lengths for masking padding
            channel_last: If True, input is (B, L, C)
            capture_layers: Set of layer indices to capture.

        Returns:
            If capture_layers: dict mapping "layer_{i}" to (B, H) tensors.
            Otherwise: (B, H) tensor from final layer.
        """
        out, intermediates = self.forward(
            x, channel_last=channel_last, capture_layers=capture_layers,
        )

        if capture_layers:
            return {
                f"layer_{i}": _mean_unpadded(intermediates[i], lengths)
                for i in sorted(intermediates)
            }
        return _mean_unpadded(out, lengths)


class OrthrusNativeEncoder(FoundationEncoder):
    """Native Orthrus encoder using mamba-ssm 2.x.

    This encoder loads Orthrus weights from HuggingFace and runs inference
    using mamba-ssm 2.x, avoiding the version conflict with Evo2.

    Args:
        model_name: HuggingFace model ID or local path.
            Default: "quietflamingo/orthrus-base-4-track"
        device: Device for inference ("cuda" or "cpu").

    Example:
        >>> encoder = OrthrusNativeEncoder()
        >>> embedding = encoder.encode("AUGCAUGCAUGCAUGC")
        >>> print(embedding.shape)  # torch.Size([1, 256])
    """

    # Model configs from HuggingFace
    MODEL_CONFIGS = {
        "quietflamingo/orthrus-base-4-track": {
            "d_model": 256,
            "n_layer": 8,
            "input_dim": 4,
        },
        "quietflamingo/orthrus-large-6-track": {
            "d_model": 512,
            "n_layer": 12,
            "input_dim": 6,
        },
    }

    def __init__(
        self,
        model_name: str = "quietflamingo/orthrus-base-4-track",
        device: str = "cuda",
        layer_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self.model_name = model_name

        # Get config
        if model_name in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[model_name]
        else:
            # Assume base 4-track config for unknown models
            config = self.MODEL_CONFIGS["quietflamingo/orthrus-base-4-track"]

        self._embedding_dim = config["d_model"]
        self._config = config
        self._layer_indices = layer_indices
        self._multi_layer = layer_indices is not None and len(layer_indices) > 0

    def _load_model(self) -> None:
        """Load Orthrus model from HuggingFace."""
        from huggingface_hub import hf_hub_download

        print(f"Loading Orthrus model: {self.model_name}")

        # Create model
        self._model = MixerModel(
            d_model=self._config["d_model"],
            n_layer=self._config["n_layer"],
            input_dim=self._config["input_dim"],
        )

        # Download and load weights
        try:
            weights_path = hf_hub_download(
                repo_id=self.model_name,
                filename="pytorch_model.bin",
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

            # Filter state dict to only MixerModel keys
            # Orthrus saves full ContrastiveLearningModel, we only need backbone
            model_keys = {k for k in state_dict.keys() if k.startswith("model.")}
            if model_keys:
                # Strip "model." prefix
                state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}

            self._model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded weights from {weights_path}")
        except Exception as e:
            print(f"  Warning: Could not load pretrained weights: {e}")
            print(f"  Using randomly initialized model")

        self._model = self._model.to(self.device).eval()

    @property
    def multi_layer(self) -> bool:
        return self._multi_layer

    @property
    def layer_indices(self) -> Optional[List[int]]:
        return self._layer_indices

    def encode(self, sequence: str) -> Union[Tensor, Dict[str, Tensor]]:
        """Encode an RNA sequence.

        Args:
            sequence: RNA sequence (A, C, G, U). T will be converted to U.

        Returns:
            If multi_layer: dict mapping "layer_{i}" to (1, embedding_dim) tensors.
            Otherwise: (1, embedding_dim) tensor.
        """
        self._ensure_loaded()

        x = _one_hot_encode(sequence, device=self.device)
        lengths = torch.tensor([len(sequence)], device=self.device)

        with torch.no_grad():
            if self._multi_layer:
                return self._model.representation(
                    x, lengths,
                    capture_layers=set(self._layer_indices),
                )
            else:
                return self._model.representation(x, lengths)

    @property
    def modality(self) -> str:
        return "rna"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
