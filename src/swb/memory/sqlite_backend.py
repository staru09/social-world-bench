"""Baseline backend: substring search over the run's state.db. No external service.

This is the default and it needs no ingest step — the trajectory already lives in
state.db from the rollout. It exists as a `MemoryBackend` so the sqlite path and the
real-memory-service paths go through the exact same seam, giving a free (if currently
weak) baseline to diff mem0/honcho/supermemory against.
"""

from __future__ import annotations

from pathlib import Path

from swb.memory.base import MemoryBackend, Retrieved
from swb.schema import Event
from swb.world.state import WorldState


class SqliteBackend(MemoryBackend):
    name = "sqlite"

    def __init__(self, state: WorldState):
        self.state = state

    @classmethod
    def from_handle(cls, handle: dict, run_dir=None) -> "SqliteBackend":
        if run_dir is None:
            raise ValueError("sqlite backend needs run_dir to locate state.db")
        return cls(WorldState(Path(run_dir) / "state.db"))

    def ingest(self, events: list[Event]) -> dict:
        # Data already lives in state.db; nothing to write.
        return {"backend": self.name, "namespace": None, "n_events": len(events)}

    def search(self, query: str, limit: int = 10) -> list[Retrieved]:
        return [
            Retrieved(content=e.content, step=e.step, speaker=e.speaker)
            for e in self.state.search_events(query, limit=limit)
        ]
