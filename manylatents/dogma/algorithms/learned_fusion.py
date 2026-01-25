"""Learned fusion methods for multi-channel embeddings.

Neural network-based fusion methods that learn a compressed representation
from concatenated multi-channel embeddings. These methods can capture
non-linear relationships between modalities.

Methods:
    - AutoencoderFusion: Standard bottleneck autoencoder
    - FrobeniusAEFusion: Autoencoder with Jacobian penalty (MBYL-style)

Example:
    >>> from manylatents.dogma.algorithms import AutoencoderFusion
    >>> fusion = AutoencoderFusion(
    ...     embeddings={"dna": dna_emb, "protein": prot_emb},
    ...     target_dim=128,
    ...     hidden_dims=[512, 256],
    ... )
    >>> fused = fusion.fit_transform(dummy)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from manylatents.algorithms.latent.latent_module_base import LatentModule


@dataclass
class LearnedFusionLoadings:
    """Interpretability info for learned fusion methods.

    Attributes:
        channel_ranges: Dict mapping channel name to (start, end) in concat.
        encoder_weights: First layer weights of encoder (input_dim, hidden_dim).
        channel_contributions: Dict mapping channel → submatrix of first layer.
        reconstruction_error: Final reconstruction loss on training data.
    """

    channel_ranges: Dict[str, tuple]
    encoder_weights: np.ndarray
    channel_contributions: Dict[str, np.ndarray]
    reconstruction_error: float


class Autoencoder(nn.Module):
    """Simple symmetric autoencoder with configurable architecture."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim

        # Default hidden dimensions
        if hidden_dims is None:
            # Simple 2-layer encoder: input → hidden → bottleneck
            hidden_dims = [max(target_dim * 2, input_dim // 2)]

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, target_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (mirror of encoder)
        decoder_layers = []
        prev_dim = target_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def _get_activation(self, activation: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (reconstruction, latent)."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoencoderFusion(LatentModule):
    """Learn fused representation via autoencoder bottleneck.

    Trains an autoencoder on concatenated multi-channel embeddings to learn
    a compressed representation that preserves information from all channels.

    Args:
        embeddings: Dict mapping channel names to tensors.
        target_dim: Bottleneck dimension (output dimension).
        hidden_dims: Hidden layer dimensions for encoder/decoder.
            Default: [input_dim // 2].
        dropout: Dropout rate during training.
        activation: Activation function ('relu', 'gelu', 'tanh', 'silu').
        lr: Learning rate for Adam optimizer.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        normalize: L2-normalize each channel before concatenation.
        device: Device for training ('cpu', 'cuda', or None for auto).
        n_components: Expected output dimension. Auto-computed if None.
        **kwargs: Passed to LatentModule (datamodule, init_seed, etc.)

    Attributes:
        model_: Trained autoencoder model (after fit).
        training_loss_: Final training loss.
        channel_dims: Dict of channel dimensions.
    """

    def __init__(
        self,
        embeddings: Optional[Dict[str, Union[Tensor, np.ndarray]]] = None,
        target_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 256,
        normalize: bool = False,
        device: Optional[str] = None,
        n_components: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(n_components=n_components or target_dim, **kwargs)

        # Convert numpy to torch if needed
        if embeddings is not None:
            self._embeddings = {
                k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v.float()
                for k, v in embeddings.items()
            }
        else:
            self._embeddings = None

        self._target_dim = target_dim
        self._hidden_dims = hidden_dims
        self._dropout = dropout
        self._activation = activation
        self._lr = lr
        self._epochs = epochs
        self._batch_size = batch_size
        self._normalize = normalize
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Set after fit
        self.model_: Optional[Autoencoder] = None
        self.training_loss_: float = float("inf")
        self.channel_dims: Dict[str, int] = {}
        self._channel_ranges: Dict[str, tuple] = {}

    def _get_embeddings(self) -> Dict[str, Tensor]:
        """Get embeddings from in-memory dict or datamodule."""
        if self._embeddings is not None:
            return self._embeddings

        if self.datamodule is None:
            raise ValueError(
                "AutoencoderFusion requires either `embeddings` dict or "
                "`datamodule` with get_embeddings() method."
            )

        if not hasattr(self.datamodule, "get_embeddings"):
            raise ValueError(
                f"Datamodule {type(self.datamodule).__name__} has no get_embeddings()."
            )

        return self.datamodule.get_embeddings()

    def _prepare_data(self) -> Tuple[Tensor, List[str]]:
        """Concatenate embeddings and record channel info."""
        all_embeddings = self._get_embeddings()
        channels = list(all_embeddings.keys())

        embeddings_list = []
        offset = 0
        for ch in channels:
            emb = all_embeddings[ch]
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb).float()
            dim = emb.shape[-1]
            self.channel_dims[ch] = dim
            self._channel_ranges[ch] = (offset, offset + dim)
            offset += dim

            if self._normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            embeddings_list.append(emb)

        concatenated = torch.cat(embeddings_list, dim=-1)
        return concatenated, channels

    def fit(self, x: Tensor) -> None:
        """Train autoencoder on concatenated embeddings.

        Args:
            x: Input tensor (ignored - embeddings from __init__ or datamodule)
        """
        torch.manual_seed(self.init_seed)
        np.random.seed(self.init_seed)

        concatenated, _ = self._prepare_data()
        input_dim = concatenated.shape[-1]

        # Create model
        self.model_ = Autoencoder(
            input_dim=input_dim,
            target_dim=self._target_dim,
            hidden_dims=self._hidden_dims,
            dropout=self._dropout,
            activation=self._activation,
        ).to(self._device)

        # Create dataloader
        dataset = TensorDataset(concatenated)
        loader = DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True, drop_last=False
        )

        # Train
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self._lr)
        criterion = nn.MSELoss()

        self.model_.train()
        for epoch in range(self._epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self._device)
                optimizer.zero_grad()
                recon, _ = self.model_(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            epoch_loss /= len(concatenated)

        self.training_loss_ = epoch_loss
        self.model_.eval()
        self._is_fitted = True

    def transform(self, x: Tensor) -> Tensor:
        """Encode concatenated embeddings to bottleneck representation.

        Args:
            x: Input tensor (ignored - embeddings from __init__ or datamodule)

        Returns:
            Fused embeddings of shape (N, target_dim)
        """
        if self.model_ is None:
            raise RuntimeError("Must call fit() before transform()")

        concatenated, _ = self._prepare_data()
        concatenated = concatenated.to(self._device)

        with torch.no_grad():
            latent = self.model_.encode(concatenated)

        return latent.cpu()

    def get_loadings(self) -> LearnedFusionLoadings:
        """Get interpretability info about channel contributions.

        Returns first-layer weights as a proxy for input feature importance.
        """
        if self.model_ is None:
            raise RuntimeError("Must call fit() before get_loadings()")

        # Get first encoder layer weights
        first_layer = self.model_.encoder[0]
        weights = first_layer.weight.detach().cpu().numpy()  # (hidden, input)
        weights = weights.T  # (input, hidden)

        # Extract per-channel contributions
        channel_contributions = {}
        for ch, (start, end) in self._channel_ranges.items():
            channel_contributions[ch] = weights[start:end, :]

        return LearnedFusionLoadings(
            channel_ranges=self._channel_ranges.copy(),
            encoder_weights=weights,
            channel_contributions=channel_contributions,
            reconstruction_error=self.training_loss_,
        )

    def channel_importance(self) -> Dict[str, float]:
        """Compute relative importance of each channel.

        Importance is measured as the Frobenius norm of each channel's
        contribution to the first encoder layer.
        """
        loadings = self.get_loadings()
        norms = {
            ch: np.linalg.norm(contrib, "fro")
            for ch, contrib in loadings.channel_contributions.items()
        }
        total = sum(norms.values())
        return {ch: norm / total for ch, norm in norms.items()}

    def __repr__(self) -> str:
        return (
            f"AutoencoderFusion(target_dim={self._target_dim}, "
            f"hidden_dims={self._hidden_dims}, epochs={self._epochs})"
        )


class FrobeniusAEFusion(AutoencoderFusion):
    """Autoencoder fusion with Frobenius norm Jacobian penalty (MBYL-style).

    Adds a regularization term that penalizes large Jacobian norms,
    encouraging the encoder to learn a smooth, locally-linear mapping.
    This is inspired by the MBYL paper's approach to preserving local geometry.

    The Jacobian penalty is computed via:
        ||J||_F^2 = sum_i ||d(f(x))/dx_i||^2

    Using the random projection estimator for efficiency:
        E[||J v||^2] = ||J||_F^2 / d  for v ~ N(0, I/d)

    Args:
        All args from AutoencoderFusion, plus:
        jacobian_weight: Weight for Jacobian penalty term.
        n_jacobian_samples: Number of random projections per batch.

    Example:
        >>> fusion = FrobeniusAEFusion(
        ...     embeddings={"dna": dna_emb, "protein": prot_emb},
        ...     target_dim=128,
        ...     jacobian_weight=0.1,
        ... )
        >>> fused = fusion.fit_transform(dummy)
    """

    def __init__(
        self,
        jacobian_weight: float = 0.1,
        n_jacobian_samples: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._jacobian_weight = jacobian_weight
        self._n_jacobian_samples = n_jacobian_samples

    def fit(self, x: Tensor) -> None:
        """Train autoencoder with Jacobian penalty.

        Args:
            x: Input tensor (ignored - embeddings from __init__ or datamodule)
        """
        torch.manual_seed(self.init_seed)
        np.random.seed(self.init_seed)

        concatenated, _ = self._prepare_data()
        input_dim = concatenated.shape[-1]

        # Create model
        self.model_ = Autoencoder(
            input_dim=input_dim,
            target_dim=self._target_dim,
            hidden_dims=self._hidden_dims,
            dropout=self._dropout,
            activation=self._activation,
        ).to(self._device)

        # Create dataloader
        dataset = TensorDataset(concatenated)
        loader = DataLoader(
            dataset, batch_size=self._batch_size, shuffle=True, drop_last=False
        )

        # Train with Jacobian penalty
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self._lr)
        recon_criterion = nn.MSELoss()

        self.model_.train()
        for epoch in range(self._epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self._device)
                batch.requires_grad_(True)

                optimizer.zero_grad()

                # Forward pass
                recon, latent = self.model_(batch)
                recon_loss = recon_criterion(recon, batch)

                # Compute Jacobian penalty via random projection
                jacobian_loss = self._jacobian_penalty(batch, latent)

                # Total loss
                loss = recon_loss + self._jacobian_weight * jacobian_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.shape[0]

            epoch_loss /= len(concatenated)

        self.training_loss_ = epoch_loss
        self.model_.eval()
        self._is_fitted = True

    def _jacobian_penalty(self, x: Tensor, z: Tensor) -> Tensor:
        """Compute Frobenius norm of encoder Jacobian via random projection.

        Uses the identity: E[||J v||^2] = ||J||_F^2 / d for v ~ N(0, I/d)

        Args:
            x: Input tensor (batch, input_dim)
            z: Latent tensor (batch, target_dim)

        Returns:
            Scalar tensor with estimated ||J||_F^2
        """
        batch_size, latent_dim = z.shape

        total_penalty = 0.0
        for _ in range(self._n_jacobian_samples):
            # Random projection vector
            v = torch.randn(batch_size, latent_dim, device=z.device)
            v = v / (latent_dim ** 0.5)  # Scale for unbiased estimator

            # Compute J^T v via backward pass
            (jvp,) = torch.autograd.grad(
                outputs=z,
                inputs=x,
                grad_outputs=v,
                create_graph=True,
                retain_graph=True,
            )

            # ||J v||^2 estimates ||J||_F^2 / d
            penalty = (jvp ** 2).sum() / batch_size
            total_penalty = total_penalty + penalty * latent_dim

        return total_penalty / self._n_jacobian_samples

    def __repr__(self) -> str:
        return (
            f"FrobeniusAEFusion(target_dim={self._target_dim}, "
            f"jacobian_weight={self._jacobian_weight}, epochs={self._epochs})"
        )
