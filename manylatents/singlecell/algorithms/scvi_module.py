"""scVI wrapper conforming to manylatents LightningModule contract.

Wraps scvi-tools' VAE module into the same interface as Reconstruction:
    - setup(): configures the scVI VAE from data dimensions
    - forward(x): reconstruction through decoder
    - encode(x): latent mean (qzm) for downstream use
    - training_step/validation_step: ELBO training with KL warmup

This is a port/stub — not all scVI features are exposed yet.
Batch correction, library size modeling, and scArches transfer
are deferred to post-submission integration.

Usage via Hydra config:
    algorithms/latent: scvi
    # which points at a config with _target_: manylatents.singlecell.algorithms.SCVIModule
"""

import logging

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule

logger = logging.getLogger(__name__)


class SCVIEncoder(nn.Module):
    """Encoder: expression -> latent distribution parameters (mu, logvar)."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.mean_encoder = nn.Linear(prev, latent_dim)
        self.var_encoder = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mean_encoder(h), self.var_encoder(h)


class SCVIDecoder(nn.Module):
    """Decoder: latent -> gene expression parameters.

    Outputs scale (normalized mean) which is multiplied by library size
    to get the rate parameter for the negative binomial.
    """

    def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.decoder = nn.Sequential(*layers)
        self.scale_decoder = nn.Sequential(nn.Linear(prev, output_dim), nn.Softmax(dim=-1))
        # per-gene log-dispersion
        self.px_r = nn.Parameter(torch.randn(output_dim))

    def forward(self, z: torch.Tensor, library: torch.Tensor) -> dict:
        h = self.decoder(z)
        px_scale = self.scale_decoder(h)
        px_rate = torch.exp(library) * px_scale
        px_r = torch.exp(self.px_r)
        return {"px_scale": px_scale, "px_rate": px_rate, "px_r": px_r}


class SCVIModule(LightningModule):
    """scVI-style VAE for single-cell data.

    Conforms to the manylatents LightningModule contract:
        - encode(x) -> latent tensor (for execute_step embedding extraction)
        - training_step/validation_step (for Lightning trainer)
        - setup() infers input_dim from datamodule

    The ZINB reconstruction loss and KL divergence follow Lopez et al. 2018.
    """

    def __init__(
        self,
        datamodule,
        n_latent: int = 30,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        kl_weight: float = 1.0,
        n_epochs_kl_warmup: int = 400,
        init_seed: int = 42,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.kl_weight = kl_weight
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.init_seed = init_seed

        self.save_hyperparameters(ignore=["datamodule"])
        self.encoder: SCVIEncoder | None = None
        self.decoder: SCVIDecoder | None = None

    def setup(self, stage=None):
        if self.encoder is not None:
            return
        first_batch = next(iter(self.datamodule.train_dataloader()))["data"]
        input_dim = first_batch.shape[1]

        torch.manual_seed(self.init_seed)
        hidden_dims = [self.n_hidden] * self.n_layers
        self.encoder = SCVIEncoder(input_dim, hidden_dims, self.n_latent, self.dropout)
        self.decoder = SCVIDecoder(self.n_latent, hidden_dims, input_dim, self.dropout)
        logger.info(f"SCVIModule configured: input_dim={input_dim}, n_latent={self.n_latent}")

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _get_library(self, x: torch.Tensor) -> torch.Tensor:
        """Observed log-library size (sum of counts per cell)."""
        return torch.log(x.sum(dim=-1, keepdim=True) + 1e-6)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent mean — the deterministic embedding for downstream use."""
        x_log = torch.log1p(x)
        mu, _ = self.encoder(x_log)
        return mu

    def forward(self, x: torch.Tensor) -> dict:
        x_log = torch.log1p(x)
        library = self._get_library(x)
        mu, logvar = self.encoder(x_log)
        z = self._reparameterize(mu, logvar)
        dec_out = self.decoder(z, library)
        return {"mu": mu, "logvar": logvar, "z": z, **dec_out}

    def _nb_loss(self, x: torch.Tensor, px_rate: torch.Tensor,
                 px_r: torch.Tensor) -> torch.Tensor:
        """Negative binomial log-likelihood loss."""
        eps = 1e-8
        log_theta_mu_eps = torch.log(px_r + px_rate + eps)
        ll = (
            torch.lgamma(x + px_r)
            - torch.lgamma(px_r)
            - torch.lgamma(x + 1)
            + px_r * (torch.log(px_r + eps) - log_theta_mu_eps)
            + x * (torch.log(px_rate + eps) - log_theta_mu_eps)
        )
        return -ll.sum(dim=-1).mean()

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,1))."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    def _kl_weight_for_epoch(self) -> float:
        if self.n_epochs_kl_warmup == 0:
            return self.kl_weight
        return self.kl_weight * min(1.0, self.current_epoch / self.n_epochs_kl_warmup)

    def shared_step(self, batch, batch_idx, phase: str) -> dict:
        x = batch["data"]
        out = self.forward(x)

        recon_loss = self._nb_loss(x, out["px_rate"], out["px_r"])
        kl_loss = self._kl_divergence(out["mu"], out["logvar"])
        kl_w = self._kl_weight_for_epoch()
        loss = recon_loss + kl_w * kl_loss

        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_recon", recon_loss, on_step=False, on_epoch=True)
        self.log(f"{phase}_kl", kl_loss, on_step=False, on_epoch=True)
        self.log(f"{phase}_kl_weight", kl_w, on_step=False, on_epoch=True)
        return {"loss": loss}

    def training_step(self, batch, batch_idx) -> dict:
        return self.shared_step(batch, batch_idx, phase="train")

    def validation_step(self, batch, batch_idx) -> dict:
        return self.shared_step(batch, batch_idx, phase="val")

    def test_step(self, batch, batch_idx) -> dict:
        out = self.shared_step(batch, batch_idx, phase="test")
        self.log("test_loss", out["loss"], prog_bar=True, on_epoch=True)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
