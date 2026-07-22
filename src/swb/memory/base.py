"""Memory backend interface + shared helpers.

The eval agent retrieves through exactly one seam — `ToolBox`'s `search_events` — and a
`MemoryBackend` is what sits behind it. `sqlite` is the built-in baseline (a substring
search over the run's state.db); mem0 / honcho / supermemory ingest the trajectory into a
real memory service and retrieve through its native semantic search.

The invariant that makes them interchangeable: `search()` returns `Retrieved` items that
each carry the originating `step`, and that step is what trajectory verification scores
on. Every backend attaches `{"step": N}` metadata at ingest and reads it back at search.
A hit the backend can't map to a source step gets `step=None` — shown to the agent, but it
can't count as a trajectory hit.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from swb.schema import Event


class MemoryConfigError(RuntimeError):
    """Missing API key, missing SDK, or an unknown backend name."""


@dataclass
class Retrieved:
    """One search hit, normalized across backends."""

    content: str
    step: int | None = None
    speaker: str | None = None
    score: float | None = None


class MemoryBackend(ABC):
    """A pluggable place to put the trajectory and search it back out."""

    name: str

    @abstractmethod
    def ingest(self, events: list[Event]) -> dict:
        """Write the trajectory into the backend. Returns a handle dict to persist."""

    @classmethod
    @abstractmethod
    def from_handle(cls, handle: dict, run_dir=None) -> "MemoryBackend":
        """Reconnect to an already-ingested store for eval, without re-ingesting."""

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[Retrieved]:
        """Retrieve up to `limit` hits for the query."""


def require_key(env_var: str, backend: str) -> str:
    """Return the API key for `backend`, or raise a clean error naming the env var."""
    key = os.environ.get(env_var)
    if not key:
        raise MemoryConfigError(f"{env_var} is not set; required for --memory {backend}")
    return key
