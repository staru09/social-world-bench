"""Hosted mem0 backend (MemoryClient).

Each event is stored raw with `infer=False` — mem0's default `infer=True` runs an LLM
that fuses/summarizes turns, which would destroy the 1:1 step mapping trajectory
verification needs. We namespace one run's data with `run_id` and read the `step` back
off each search hit's `metadata`.
"""

from __future__ import annotations

from swb.memory.base import MemoryBackend, MemoryConfigError, Retrieved, require_key
from swb.schema import Event


class Mem0Backend(MemoryBackend):
    name = "mem0"

    def __init__(self, namespace: str):
        self.namespace = namespace
        require_key("MEM0_API_KEY", self.name)
        try:
            from mem0 import MemoryClient
        except ImportError as e:  # pragma: no cover - exercised via monkeypatch
            raise MemoryConfigError(
                "mem0ai not installed; `pip install social-world-bench[mem0]`"
            ) from e
        self.client = MemoryClient()  # reads MEM0_API_KEY

    @classmethod
    def from_handle(cls, handle: dict, run_dir=None) -> "Mem0Backend":
        return cls(handle["namespace"])

    def ingest(self, events: list[Event]) -> dict:
        for e in events:
            self.client.add(
                [{"role": "user", "content": f"{e.speaker}: {e.content}"}],
                run_id=self.namespace,
                metadata={"step": e.step, "speaker": e.speaker},
                infer=False,
            )
        return {"backend": self.name, "namespace": self.namespace, "n_events": len(events)}

    def search(self, query: str, limit: int = 10) -> list[Retrieved]:
        resp = self.client.search(
            query, filters={"AND": [{"run_id": self.namespace}]}, top_k=limit
        )
        results = resp.get("results", resp) if isinstance(resp, dict) else resp
        return [_to_retrieved(r) for r in results]


def _to_retrieved(r: dict) -> Retrieved:
    meta = r.get("metadata") or {}
    step = meta.get("step")
    return Retrieved(
        content=r.get("memory", ""),
        step=int(step) if step is not None else None,
        speaker=meta.get("speaker"),
        score=r.get("score"),
    )
