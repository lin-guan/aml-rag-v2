from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str,
        device: str | None,
        batch_size: int,
        max_concurrent_encodes: int,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._model_lock = threading.Lock()
        self._encode_semaphore = asyncio.Semaphore(max_concurrent_encodes)

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is None:
                kwargs = {"device": self._device} if self._device else {}
                self._model = SentenceTransformer(self._model_name, **kwargs)

    def _encode_sync(self, texts: Sequence[str]) -> np.ndarray:
        self.load()
        assert self._model is not None
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    async def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        async with self._encode_semaphore:
            return await asyncio.to_thread(self._encode_sync, texts)
