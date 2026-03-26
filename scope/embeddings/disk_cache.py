"""Disk-based embedding cache using numpy .npz files."""

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DiskEmbeddingCache:
    """Persistent embedding cache backed by numpy .npz files.

    Storage layout:
        {cache_dir}/{model_hash}.npz      — hash→embedding mapping
        {cache_dir}/{model_hash}.keys.json — hash→original text (debug)
    """

    def __init__(self, cache_dir: str, model_name: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._model_name = model_name
        self._file_stem = self._safe_filename(model_name)
        self._npz_path = self._cache_dir / f"{self._file_stem}.npz"
        self._keys_path = self._cache_dir / f"{self._file_stem}.keys.json"

        # In-memory dict: hash_key → np.ndarray
        self._data: dict[str, np.ndarray] = {}
        # Reverse lookup: hash_key → original text
        self._key_map: dict[str, str] = {}
        self._dirty = False
        self._hits = 0
        self._misses = 0

        self._load()

    @staticmethod
    def _safe_filename(model_name: str) -> str:
        return model_name.replace("/", "_").replace("\\", "_")

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self._npz_path.exists():
            npz = np.load(self._npz_path)
            for key in npz.files:
                self._data[key] = npz[key]
            logger.info(f"Loaded {len(self._data)} cached embeddings from {self._npz_path}")

        if self._keys_path.exists():
            with open(self._keys_path) as f:
                self._key_map = json.load(f)

    def get(self, text: str) -> Optional[np.ndarray]:
        h = self._hash_text(text)
        if h in self._data:
            self._hits += 1
            return self._data[h]
        self._misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        h = self._hash_text(text)
        self._data[h] = embedding
        self._key_map[h] = text
        self._dirty = True

    def get_batch(self, texts: list[str]) -> dict[str, Optional[np.ndarray]]:
        return {text: self.get(text) for text in texts}

    def put_batch(self, items: dict[str, np.ndarray]) -> None:
        for text, embedding in items.items():
            self.put(text, embedding)

    def flush(self) -> None:
        if not self._dirty:
            return
        # Atomic write: write to temp file then rename to avoid corruption on interrupt
        # np.savez appends .npz if not present, so we use a stem path and track the actual file
        tmp_dir = tempfile.mkdtemp(dir=self._cache_dir)
        tmp_stem = Path(tmp_dir) / "cache"
        try:
            np.savez(tmp_stem, **self._data)
            actual_tmp = Path(str(tmp_stem) + ".npz")
            actual_tmp.rename(self._npz_path)
        except Exception:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        else:
            Path(tmp_dir).rmdir()
        with open(self._keys_path, "w") as f:
            json.dump(self._key_map, f)
        logger.info(
            f"Flushed {len(self._data)} embeddings to {self._npz_path} "
            f"(hits={self._hits}, misses={self._misses})"
        )
        self._dirty = False

    def stats(self) -> dict:
        return {
            "size": len(self._data),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1),
            "path": str(self._npz_path),
        }
