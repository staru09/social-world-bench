"""Pluggable memory backends for the eval stage.

`get_backend(name)` returns a backend class; each SDK is imported lazily inside its module
so a missing extra only errors when that backend is actually used.
"""

from __future__ import annotations

from swb.memory.base import MemoryBackend, MemoryConfigError, Retrieved

BACKEND_NAMES = ["sqlite", "mem0", "honcho", "supermemory"]


def get_backend(name: str) -> type[MemoryBackend]:
    if name == "sqlite":
        from swb.memory.sqlite_backend import SqliteBackend

        return SqliteBackend
    if name == "mem0":
        from swb.memory.mem0_backend import Mem0Backend

        return Mem0Backend
    if name == "honcho":
        from swb.memory.honcho_backend import HonchoBackend

        return HonchoBackend
    if name == "supermemory":
        from swb.memory.supermemory_backend import SupermemoryBackend

        return SupermemoryBackend
    raise MemoryConfigError(f"unknown memory backend {name!r}; expected one of {BACKEND_NAMES}")


__all__ = ["MemoryBackend", "MemoryConfigError", "Retrieved", "get_backend", "BACKEND_NAMES"]
