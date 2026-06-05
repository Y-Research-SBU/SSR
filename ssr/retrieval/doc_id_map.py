"""Map external corpus ids (often strings) to dense int32 doc indices for the index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


@dataclass
class DocIdMap:
    """Bijection between external doc ids and ``0 .. n_docs-1`` for inverted-index storage."""

    external_ids: list[str]

    def __init__(self, external_ids: list[str] | None = None) -> None:
        self.external_ids = list(external_ids) if external_ids is not None else []
        self._ext_to_int: dict[str, int] = {
            ext: i for i, ext in enumerate(self.external_ids)
        }

    @property
    def n_docs(self) -> int:
        return len(self.external_ids)

    def assign(self, external_id: str) -> int:
        ext = str(external_id)
        i = self._ext_to_int.get(ext)
        if i is None:
            i = len(self.external_ids)
            self._ext_to_int[ext] = i
            self.external_ids.append(ext)
        return i

    def assign_batch(self, external_ids: Sequence[str]) -> np.ndarray:
        return np.asarray([self.assign(x) for x in external_ids], dtype=np.int32)

    def global_starts_for_batch(self, external_ids: Sequence[str]) -> tuple[np.ndarray, int]:
        """Return per-doc global int ids and ``min`` id (batch global doc start)."""
        gids = self.assign_batch(external_ids)
        return gids, int(gids.min()) if gids.size else 0

    def external_id(self, doc_idx: int) -> str:
        return self.external_ids[int(doc_idx)]

    def to_json(self) -> dict:
        return {
            "version": 1,
            "n_docs": self.n_docs,
            "external_ids": self.external_ids,
        }

    @classmethod
    def from_json(cls, payload: dict) -> DocIdMap:
        if int(payload.get("version", 0)) != 1:
            raise ValueError(f"Unsupported doc_id_map version: {payload.get('version')}")
        return cls(list(payload["external_ids"]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> DocIdMap:
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def iter_pairs(self) -> Iterator[tuple[int, str]]:
        for i, ext in enumerate(self.external_ids):
            yield i, ext
