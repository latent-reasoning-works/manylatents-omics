"""scVI wrapper conforming to manylatents LightningModule contract.

Wraps scvi-tools' VAE module into the same interface as Reconstruction:
    - setup(): configures the scVI VAE from data dimensions
    - forward(x): reconstruction through decoder
    - encode(x): latent mean (qzm) for downstream use
    - training_step/validation_step: ELBO training with KL warmup

Implements the full scVI generative model (Lopez et al. 2018):
    - ZINB/NB gene likelihood with learned per-gene dispersion
    - KL warmup schedule
    - Batch correction via learned batch embeddings
    - Optional learned library size encoder

Usage via Hydra config:
    algorithms/latent: scvi
    # which points at a config with _target_: manylatents.singlecell.algorithms.SCVIModule
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SCVILibraryEncoder(nn.Module):
    """Learned library size encoder: expression -> log-library distribution (mu, logvar)."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_encoder = nn.Linear(hidden_dim, 1)
        self.var_encoder = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mean_encoder(h), self.var_encoder(h)


class SCVIDecoder(nn.Module):
    """Decoder: latent -> gene expression parameters.

    Outputs scale (normalized mean) which is multiplied by library size
    to get the rate parameter for the negative binomial. Optionally outputs
    dropout logits for zero-inflation (ZINB).
    """

    def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int,
                 n_batch: int = 0, dropout: float = 0.1, zero_inflated: bool = True):
        super().__init__()
        self.zero_inflated = zero_inflated
        # Batch embedding concatenated to latent input
        self.batch_embedding = nn.Embedding(n_batch, latent_dim) if n_batch > 0 else None
        decoder_input_dim = latent_dim * 2 if n_batch > 0 else latent_dim

        layers = []
        prev = decoder_input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.decoder = nn.Sequential(*layers)
        self.scale_decoder = nn.Sequential(nn.Linear(prev, output_dim), nn.Softmax(dim=-1))
        # Per-gene log-dispersion (inverse overdispersion)
        self.px_r = nn.Parameter(torch.randn(output_dim))
        # Zero-inflation logits
        if zero_inflated:
            self.px_dropout = nn.Linear(prev, output_dim)

    def forward(self, z: torch.Tensor, library: torch.Tensor,
                batch_index: torch.Tensor | None = None) -> dict:
        if self.batch_embedding is not None and batch_index is not None:
            batch_emb = self.batch_embedding(batch_index.long().squeeze(-1))
            z = torch.cat([z, batch_emb], dim=-1)

        h = self.decoder(z)
        px_scale = self.scale_decoder(h)
        px_rate = torch.exp(library) * px_scale
        px_r = torch.exp(self.px_r)
        out = {"px_scale": px_scale, "px_rate": px_rate, "px_r": px_r}
        if self.zero_inflated:
            out["px_dropout"] = self.px_dropout(h)
        return out


class SCVIModule(LightningModule):
    """scVI-style VAE for single-cell data.

    Conforms to the manylatents LightningModule contract:
        - encode(x) -> latent tensor (for execute_step embedding extraction)
        - training_step/validation_step (for Lightning trainer)
        - setup() infers input_dim from datamodule

    Implements the full ZINB generative model from Lopez et al. 2018.
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
        gene_likelihood: str = "zinb",
        n_batch: int = 0,
        use_observed_lib_size: bool = True,
        init_seed: int = 42,
    ):
        """
        Args:
            datamodule: LightningDataModule providing {"data", "metadata"} batches.
            n_latent: Latent space dimensionality.
            n_hidden: Hidden layer width.
            n_layers: Number of hidden layers in encoder/decoder.
            dropout: Dropout rate.
            lr: Learning rate.
            kl_weight: Maximum KL weight (reached after warmup).
            n_epochs_kl_warmup: Epochs to linearly ramp KL weight from 0 to kl_weight.
            gene_likelihood: "zinb" (zero-inflated NB) or "nb" (negative binomial).
            n_batch: Number of batch categories for batch correction (0 = disabled).
            use_observed_lib_size: If True, use observed log-library. If False, learn it.
            init_seed: Random seed for weight initialization.
        """
        super().__init__()
        self.datamodule = datamodule
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.kl_weight = kl_weight
        self.n_epochs_kl_warmup = n_epochs_kl_warmup
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.use_observed_lib_size = use_observed_lib_size
        self.init_seed = init_seed

        self.save_hyperparameters(ignore=["datamodule"])
        self.encoder: SCVIEncoder | None = None
        self.decoder: SCVIDecoder | None = None
        self.library_encoder: SCVILibraryEncoder | None = None

    def setup(self, stage=None):
        if self.encoder is not None:
            return
        first_batch = next(iter(self.datamodule.train_dataloader()))["data"]
        input_dim = first_batch.shape[1]

        torch.manual_seed(self.init_seed)
        hidden_dims = [self.n_hidden] * self.n_layers
        self.encoder = SCVIEncoder(input_dim, hidden_dims, self.n_latent, self.dropout)
        self.decoder = SCVIDecoder(
            self.n_latent, hidden_dims, input_dim,
            n_batch=self.n_batch, dropout=self.dropout,
            zero_inflated=(self.gene_likelihood == "zinb"),
        )
        if not self.use_observed_lib_size:
            self.library_encoder = SCVILibraryEncoder(input_dim, self.n_hidden, self.dropout)

        logger.info(
            f"SCVIModule configured: input_dim={input_dim}, n_latent={self.n_latent}, "
            f"likelihood={self.gene_likelihood}, n_batch={self.n_batch}, "
            f"learned_library={not self.use_observed_lib_size}"
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _get_library(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return (library, ql_mu, ql_logvar). Last two are None if using observed."""
        if self.use_observed_lib_size or self.library_encoder is None:
            return torch.log(x.sum(dim=-1, keepdim=True) + 1e-6), None, None
        x_log = torch.log1p(x)
        ql_mu, ql_logvar = self.library_encoder(x_log)
        library = self._reparameterize(ql_mu, ql_logvar)
        return library, ql_mu, ql_logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent mean — the deterministic embedding for downstream use."""
        x_log = torch.log1p(x)
        mu, _ = self.encoder(x_log)
        return mu

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor | None = None) -> dict:
        x_log = torch.log1p(x)
        library, ql_mu, ql_logvar = self._get_library(x)
        mu, logvar = self.encoder(x_log)
        z = self._reparameterize(mu, logvar)
        dec_out = self.decoder(z, library, batch_index)
        out = {"mu": mu, "logvar": logvar, "z": z, **dec_out}
        if ql_mu is not None:
            out["ql_mu"] = ql_mu
            out["ql_logvar"] = ql_logvar
        return out

    # --- Loss components ---

    def _nb_log_likelihood(self, x: torch.Tensor, px_rate: torch.Tensor,
                           px_r: torch.Tensor) -> torch.Tensor:
        """Per-gene negative binomial log-likelihood. Returns (batch, genes)."""
        eps = 1e-8
        log_theta_mu_eps = torch.log(px_r + px_rate + eps)
        return (
            torch.lgamma(x + px_r)
            - torch.lgamma(px_r)
            - torch.lgamma(x + 1)
            + px_r * (torch.log(px_r + eps) - log_theta_mu_eps)
            + x * (torch.log(px_rate + eps) - log_theta_mu_eps)
        )

    def _reconstruction_loss(self, x: torch.Tensor, out: dict) -> torch.Tensor:
        """ZINB or NB reconstruction loss, averaged over batch."""
        nb_ll = self._nb_log_likelihood(x, out["px_rate"], out["px_r"])

        if self.gene_likelihood == "zinb" and "px_dropout" in out:
            # ZINB: mixture of point mass at zero and NB
            px_dropout = out["px_dropout"]
            # P(zero) from NB component
            nb_zero = (out["px_r"] * (torch.log(out["px_r"] + 1e-8)
                       - torch.log(out["px_r"] + out["px_rate"] + 1e-8)))
            # log P(x | zinb)
            zero_case = torch.logsumexp(
                torch.stack([F.logsigmoid(px_dropout) + nb_zero,
                             F.logsigmoid(-px_dropout) + nb_ll], dim=0),
                dim=0,
            )
            nonzero_case = F.logsigmoid(-px_dropout) + nb_ll
            ll = torch.where(x < 0.5, zero_case, nonzero_case)
        else:
            ll = nb_ll

        return -ll.sum(dim=-1).mean()

    def _kl_z(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || N(0,1))."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    def _kl_library(self, x: torch.Tensor, ql_mu: torch.Tensor,
                    ql_logvar: torch.Tensor) -> torch.Tensor:
        """KL for learned library size against empirical prior N(mean, var) of log-library."""
        # Empirical prior: per-batch log-library statistics
        observed = torch.log(x.sum(dim=-1, keepdim=True) + 1e-6)
        pl_mu = observed.mean()
        pl_var = observed.var() + 1e-4
        return -0.5 * torch.sum(
            1 + ql_logvar - torch.log(pl_var)
            - (ql_logvar.exp() + (ql_mu - pl_mu).pow(2)) / pl_var,
            dim=-1,
        ).mean()

    def _kl_weight_for_epoch(self) -> float:
        if self.n_epochs_kl_warmup == 0:
            return self.kl_weight
        return self.kl_weight * min(1.0, self.current_epoch / self.n_epochs_kl_warmup)

    # --- Training loop ---

    def shared_step(self, batch, batch_idx, phase: str) -> dict:
        x = batch["data"]
        batch_index = batch.get("batch_index", None)
        out = self.forward(x, batch_index)

        recon_loss = self._reconstruction_loss(x, out)
        kl_z = self._kl_z(out["mu"], out["logvar"])
        kl_w = self._kl_weight_for_epoch()
        loss = recon_loss + kl_w * kl_z

        # Library KL if learned
        if "ql_mu" in out:
            kl_lib = self._kl_library(x, out["ql_mu"], out["ql_logvar"])
            loss = loss + kl_lib
            self.log(f"{phase}_kl_lib", kl_lib, on_step=False, on_epoch=True)

        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{phase}_recon", recon_loss, on_step=False, on_epoch=True)
        self.log(f"{phase}_kl_z", kl_z, on_step=False, on_epoch=True)
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
