"""Honcho backend.

Model: one peer per speaker, one session holding the whole trajectory, each message
authored by its speaker's peer and tagged with `{"step": N}` metadata. Retrieval uses
`session.search()` (hybrid full-text + semantic over the raw messages), which returns the
source `Message` objects with their metadata — NOT `peer.chat()`, which synthesizes an
answer string and would give us no step to verify against.
"""

from __future__ import annotations

from swb.memory.base import MemoryBackend, MemoryConfigError, Retrieved, require_key
from swb.schema import Event


class HonchoBackend(MemoryBackend):
    name = "honcho"

    def __init__(self, namespace: str):
        self.namespace = namespace
        require_key("HONCHO_API_KEY", self.name)
        try:
            from honcho import Honcho
        except ImportError as e:  # pragma: no cover - exercised via monkeypatch
            raise MemoryConfigError(
                "honcho-ai not installed; `pip install social-world-bench[honcho]`"
            ) from e
        self.honcho = Honcho(workspace_id=namespace)  # reads HONCHO_API_KEY
        self.session = self.honcho.session(f"{namespace}-traj")

    @classmethod
    def from_handle(cls, handle: dict, run_dir=None) -> "HonchoBackend":
        return cls(handle["namespace"])

    def ingest(self, events: list[Event]) -> dict:
        peers: dict[str, object] = {}
        for e in events:
            if e.speaker not in peers:
                peers[e.speaker] = self.honcho.peer(e.speaker)
        self.session.add_peers(list(peers.values()))
        self.session.add_messages(
            [peers[e.speaker].message(e.content, metadata={"step": e.step}) for e in events]
        )
        return {"backend": self.name, "namespace": self.namespace, "n_events": len(events)}

    def search(self, query: str, limit: int = 10) -> list[Retrieved]:
        results = self.session.search(query)
        return [_to_retrieved(m) for m in list(results)[:limit]]


def _to_retrieved(m) -> Retrieved:
    meta = getattr(m, "metadata", None) or {}
    step = meta.get("step")
    return Retrieved(
        content=getattr(m, "content", ""),
        step=int(step) if step is not None else None,
        speaker=getattr(m, "peer_id", None),
    )
