"""scGPT zero-shot cell encoder for manylatents.

Wraps the pretrained scGPT model (Cui et al., Nature Methods 2024) as a
LatentModule.  Given an AnnData-backed datamodule, it produces per-cell
embeddings using the CLS token from the frozen transformer — no fine-tuning.

The model architecture and data collator are vendored from the scGPT repo
(MIT license) under ``_scgpt_vendor/``, with the ``torchtext`` dependency
replaced by a lightweight dict-based vocabulary.

Usage via Hydra:
    algorithms/latent: scgpt
    # points at a config with _target_: manylatents.singlecell.algorithms.ScGPTEncoder

Model weights:
    Download ``scGPT_human`` from https://github.com/bowang-lab/scGPT
    and point ``model_dir`` at the unzipped directory containing
    ``best_model.pt``, ``args.json``, and ``vocab.json``.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from manylatents.algorithms.latent.latent_module_base import LatentModule

from ._scgpt_vendor.data_collator import DataCollator
from ._scgpt_vendor.model import TransformerModel
from ._scgpt_vendor.vocab import GeneVocab

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, torch.Tensor]


def _load_pretrained(
    model: torch.nn.Module,
    pretrained_params: dict,
    use_flash_attn: bool = False,
) -> torch.nn.Module:
    """Load pretrained weights (non-strict, shape-matched)."""
    if not use_flash_attn:
        pretrained_params = {
            k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_params.items()
        }
    model_dict = model.state_dict()
    matched = {
        k: v
        for k, v in pretrained_params.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    logger.info("scGPT: loaded %d/%d parameter tensors", len(matched), len(model_dict))
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    return model


class ScGPTEncoder(LatentModule):
    """Zero-shot cell embedding encoder using pretrained scGPT.

    Implements the manylatents LatentModule interface:
        - fit(): no-op (pretrained, nothing to learn)
        - transform(): loads model, encodes all cells, returns (N, embsize)

    Args:
        model_dir: Path to the scGPT_human model directory.
        gene_col: Column in ``adata.var`` containing HGNC gene names.
        max_length: Maximum number of genes per cell (longer are subsampled).
        batch_size: Inference batch size.
        device: Torch device string.
        use_fast_transformer: Whether to use flash-attention.
        n_hvg: Number of highly variable genes to select. If 0, uses all genes.
    """

    # Default weights location on Mila cluster
    DEFAULT_MODEL_DIR = "/network/weights/scGPT_human"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        gene_col: str = "gene_name",
        max_length: int = 1200,
        batch_size: int = 64,
        device: str = "cuda",
        use_fast_transformer: bool = False,
        n_hvg: int = 3000,
        **kwargs,
    ):
        # n_components is the embedding dim — set after loading model config
        super().__init__(n_components=0, **kwargs)
        self.model_dir = Path(model_dir or self.DEFAULT_MODEL_DIR)
        self.gene_col = gene_col
        self.max_length = max_length
        self.batch_size = batch_size
        self._device = device
        self.use_fast_transformer = use_fast_transformer
        self.n_hvg = n_hvg

        self._model = None
        self._vocab = None
        self._model_configs = None

    def _load_model(self):
        """Lazy load the scGPT model, vocab, and config."""
        if self._model is not None:
            return

        vocab_file = self.model_dir / "vocab.json"
        config_file = self.model_dir / "args.json"
        model_file = self.model_dir / "best_model.pt"

        for f in [vocab_file, config_file, model_file]:
            if not f.exists():
                raise FileNotFoundError(
                    f"scGPT model file not found: {f}\n"
                    f"Download scGPT_human from https://github.com/bowang-lab/scGPT"
                )

        # Vocabulary
        self._vocab = GeneVocab.from_file(vocab_file)
        for s in ["<pad>", "<cls>", "<eoc>"]:
            if s not in self._vocab:
                self._vocab.append_token(s)

        # Model config
        with open(config_file, "r") as f:
            self._model_configs = json.load(f)

        self.n_components = self._model_configs["embsize"]
        self._vocab.set_default_index(self._vocab["<pad>"])

        # Device
        device = torch.device(self._device)
        if self._device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = torch.device("cpu")

        # Build model
        self._model = TransformerModel(
            ntoken=len(self._vocab),
            d_model=self._model_configs["embsize"],
            nhead=self._model_configs["nheads"],
            d_hid=self._model_configs["d_hid"],
            nlayers=self._model_configs["nlayers"],
            nlayers_cls=self._model_configs["n_layers_cls"],
            n_cls=1,
            vocab=self._vocab,
            dropout=self._model_configs["dropout"],
            pad_token=self._model_configs["pad_token"],
            pad_value=self._model_configs["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=self.use_fast_transformer,
            fast_transformer_backend="flash",
            pre_norm=False,
        )
        _load_pretrained(
            self._model,
            torch.load(model_file, map_location=device),
            use_flash_attn=self.use_fast_transformer,
        )
        self._model.to(device).eval()
        logger.info(
            "scGPT loaded: %s, embsize=%d", self.model_dir.name, self.n_components
        )

    def _get_adata(self):
        """Extract AnnData from the datamodule."""
        if self.datamodule is None:
            raise ValueError(
                "ScGPTEncoder requires a datamodule with an AnnData-backed dataset. "
                "Pass datamodule= when instantiating."
            )
        # AnnDataModule stores the dataset which has .adata
        ds = self.datamodule
        # Walk through possible datamodule structures
        if hasattr(ds, "dataset") and hasattr(ds.dataset, "adata"):
            return ds.dataset.adata
        if hasattr(ds, "train_dataset") and hasattr(ds.train_dataset, "adata"):
            return ds.train_dataset.adata
        # Try setup() first if not already done
        if hasattr(ds, "setup"):
            ds.setup("fit")
            if hasattr(ds, "dataset") and hasattr(ds.dataset, "adata"):
                return ds.dataset.adata
            if hasattr(ds, "train_dataset") and hasattr(ds.train_dataset, "adata"):
                return ds.train_dataset.adata
        raise ValueError(
            f"Cannot extract AnnData from datamodule {type(ds).__name__}. "
            "Expected .dataset.adata or .train_dataset.adata."
        )

    def _preprocess_adata(self, adata):
        """Preprocess AnnData: HVG selection + gene-to-vocab mapping."""
        import scanpy as sc

        adata = adata.copy()

        # HVG selection if requested
        if self.n_hvg > 0 and adata.shape[1] > self.n_hvg:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=self.n_hvg, flavor="seurat_v3"
            )
            adata = adata[:, adata.var["highly_variable"]]

        # Resolve gene column
        if self.gene_col == "index":
            adata.var["index"] = adata.var.index
        elif self.gene_col not in adata.var:
            # Try var index directly
            adata.var["gene_name"] = adata.var.index

        gene_col = self.gene_col if self.gene_col in adata.var else "gene_name"

        # Map genes to vocabulary IDs
        adata.var["id_in_vocab"] = [
            self._vocab[gene] if gene in self._vocab else -1
            for gene in adata.var[gene_col]
        ]
        n_matched = (np.array(adata.var["id_in_vocab"]) >= 0).sum()
        n_total = len(adata.var)
        logger.info(
            "scGPT: matched %d/%d genes in vocabulary of size %d",
            n_matched,
            n_total,
            len(self._vocab),
        )

        # Filter to matched genes only
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        genes = adata.var[gene_col].tolist()
        gene_ids = np.array(self._vocab(genes), dtype=int)

        return adata, gene_ids

    def _embed_cells(self, adata, gene_ids: np.ndarray) -> np.ndarray:
        """Run the scGPT model to get CLS token embeddings for all cells."""
        count_matrix = adata.X
        if not isinstance(count_matrix, np.ndarray):
            count_matrix = count_matrix.toarray()

        # Inner dataset that yields per-cell sparse (nonzero genes + values)
        vocab = self._vocab
        model_configs = self._model_configs

        class _CellDataset(torch.utils.data.Dataset):
            def __init__(self, count_matrix, gene_ids):
                self.count_matrix = count_matrix
                self.gene_ids = gene_ids

            def __len__(self):
                return len(self.count_matrix)

            def __getitem__(self, idx):
                row = self.count_matrix[idx]
                nonzero_idx = np.nonzero(row)[0]
                values = row[nonzero_idx]
                genes = self.gene_ids[nonzero_idx]
                # Prepend <cls> token
                genes = np.insert(genes, 0, vocab["<cls>"])
                values = np.insert(values, 0, model_configs["pad_value"])
                return {
                    "id": idx,
                    "genes": torch.from_numpy(genes).long(),
                    "expressions": torch.from_numpy(values).float(),
                }

        dataset = _CellDataset(count_matrix, gene_ids)
        collator = DataCollator(
            do_padding=True,
            pad_token_id=vocab[model_configs["pad_token"]],
            pad_value=model_configs["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=self.max_length,
            sampling=True,
            keep_first_n_tokens=1,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=min(len(os.sched_getaffinity(0)), self.batch_size),
            pin_memory=True,
        )

        device = next(self._model.parameters()).device
        embsize = self._model_configs["embsize"]
        cell_embeddings = np.zeros((len(dataset), embsize), dtype=np.float32)

        from tqdm import tqdm

        count = 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            for data_dict in tqdm(data_loader, desc="scGPT embedding"):
                input_gene_ids = data_dict["gene"].to(device)
                src_key_padding_mask = input_gene_ids.eq(
                    vocab[self._model_configs["pad_token"]]
                )
                embeddings = self._model._encode(
                    input_gene_ids,
                    data_dict["expr"].to(device),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                )
                # CLS token is at position 0
                embeddings = embeddings[:, 0, :].cpu().numpy()
                cell_embeddings[count : count + len(embeddings)] = embeddings
                count += len(embeddings)

        # L2-normalize (same as scGPT's embed_data)
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )
        return cell_embeddings

    # --- LatentModule interface ---

    def fit(self, x: ArrayLike, y: ArrayLike = None) -> None:
        """No-op — scGPT is pretrained."""
        self._is_fitted = True

    def transform(self, x: ArrayLike) -> ArrayLike:
        """Encode all cells via scGPT and return embeddings."""
        self._load_model()
        adata = self._get_adata()
        adata, gene_ids = self._preprocess_adata(adata)
        embeddings = self._embed_cells(adata, gene_ids)
        return torch.from_numpy(embeddings) if isinstance(x, torch.Tensor) else embeddings
