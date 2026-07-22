"""Supermemory backend.

Supermemory's metadata is per-DOCUMENT, so we store one document per event (each with its
own `{"step": N}`) rather than one document for the whole trajectory. Runs are namespaced
by a container tag. Ingest is asynchronous — the API extracts/indexes in the background —
so `ingest()` polls until nothing is processing for this tag before returning, otherwise an
immediately-following eval would search an empty index.
"""

from __future__ import annotations

import time

from swb.memory.base import MemoryBackend, MemoryConfigError, Retrieved, require_key
from swb.schema import Event


class SupermemoryBackend(MemoryBackend):
    name = "supermemory"

    def __init__(self, namespace: str):
        self.namespace = namespace
        require_key("SUPERMEMORY_API_KEY", self.name)
        try:
            from supermemory import Supermemory
        except ImportError as e:  # pragma: no cover - exercised via monkeypatch
            raise MemoryConfigError(
                "supermemory not installed; `pip install social-world-bench[supermemory]`"
            ) from e
        self.client = Supermemory()  # reads SUPERMEMORY_API_KEY

    @classmethod
    def from_handle(cls, handle: dict, run_dir=None) -> "SupermemoryBackend":
        return cls(handle["namespace"])

    def ingest(self, events: list[Event]) -> dict:
        for e in events:
            self.client.add(
                content=f"{e.speaker}: {e.content}",
                container_tags=[self.namespace],
                metadata={"step": e.step, "speaker": e.speaker},
            )
        self._await_indexing()
        return {"backend": self.name, "namespace": self.namespace, "n_events": len(events)}

    def _await_indexing(self, timeout: float = 180.0, interval: float = 3.0) -> None:
        """Block until nothing is processing for this tag (async indexing lag)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            processing = self.client.documents.list_processing(container_tags=[self.namespace])
            items = getattr(processing, "results", processing)
            if not items:
                return
            time.sleep(interval)

    def search(self, query: str, limit: int = 10) -> list[Retrieved]:
        resp = self.client.search.documents(
            q=query, container_tags=[self.namespace], limit=limit
        )
        return [_to_retrieved(r) for r in getattr(resp, "results", [])]


def _to_retrieved(r) -> Retrieved:
    meta = getattr(r, "metadata", None) or {}
    step = meta.get("step")
    content = " ".join(getattr(c, "content", "") for c in getattr(r, "chunks", []))
    return Retrieved(
        content=content,
        step=int(step) if step is not None else None,
        speaker=meta.get("speaker"),
        score=getattr(r, "score", None),
    )
