# Minimal gene vocabulary — replaces torchtext-dependent GeneVocab.
# Provides the same interface used by scGPT's embedding pipeline:
#   vocab[gene_name] → int, vocab(list_of_genes) → list_of_ints,
#   len(vocab), "gene" in vocab, vocab.set_default_index(idx).

import json
from pathlib import Path
from typing import List, Union


class GeneVocab:
    """Dict-backed gene vocabulary (no torchtext dependency)."""

    def __init__(self, token2idx: dict):
        self._token2idx = dict(token2idx)
        self._default_index = None

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "GeneVocab":
        file_path = Path(file_path)
        with file_path.open("r") as f:
            token2idx = json.load(f)
        return cls(token2idx)

    def set_default_index(self, index: int) -> None:
        self._default_index = index

    def append_token(self, token: str) -> None:
        if token not in self._token2idx:
            self._token2idx[token] = len(self._token2idx)

    def __getitem__(self, token: str) -> int:
        if token in self._token2idx:
            return self._token2idx[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(token)

    def __contains__(self, token: str) -> bool:
        return token in self._token2idx

    def __len__(self) -> int:
        return len(self._token2idx)

    def __call__(self, tokens: List[str]) -> List[int]:
        return [self[t] for t in tokens]
