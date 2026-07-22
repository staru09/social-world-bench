"""Search tools exposed to the eval agent (the in-process stand-in for MCP).

The agent gets no history in its prompt; the only path to an answer is querying the
trajectory through these tools. Every call is logged with the event steps it
returned, and that log is what trajectory verification consumes.
"""

from __future__ import annotations

from swb.memory.base import MemoryBackend, Retrieved
from swb.memory.sqlite_backend import SqliteBackend
from swb.schema import ToolCall
from swb.world.state import WorldState

# OpenAI/OpenRouter tool schemas.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_events",
            "description": "Semantic search over the conversation memory. Returns matching messages with their step numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "keyword or phrase to search for"},
                    "limit": {"type": "integer", "description": "max results (default 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events_range",
            "description": "Return all messages between two step numbers, inclusive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_peer_info",
            "description": "Get a participant's profile and their relationships to others.",
            "parameters": {
                "type": "object",
                "properties": {"peer_id": {"type": "string"}},
                "required": ["peer_id"],
            },
        },
    },
]


class ToolBox:
    """Dispatches tool calls against a run's WorldState and records a query log.

    `search_events` goes through the pluggable memory `backend` (defaulting to the sqlite
    baseline); `get_events_range` / `get_peer_info` stay on `state` because they are
    structural reads of the run, independent of which memory system is under test.
    """

    def __init__(self, state: WorldState, task_id: str, backend: MemoryBackend | None = None):
        self.state = state
        self.task_id = task_id
        self.backend = backend if backend is not None else SqliteBackend(state)
        self.log: list[ToolCall] = []

    def dispatch(self, name: str, args: dict) -> str:
        if name == "search_events":
            hits = self.backend.search(args["query"], limit=int(args.get("limit", 10)))
            # Only steps we could recover count toward trajectory verification.
            self._record(name, args, [h.step for h in hits if h.step is not None])
            return self._fmt_hits(hits)
        if name == "get_events_range":
            events = self.state.get_events_range(int(args["start"]), int(args["end"]))
            self._record(name, args, [e.step for e in events])
            return self._fmt(events)
        if name == "get_peer_info":
            peer = self.state.get_peer(args["peer_id"])
            self._record(name, args, [])
            if peer is None:
                return f"No peer with id {args['peer_id']}."
            rels = [
                f"{r.type} of {r.to_peer}"
                for r in self.state.get_relationships()
                if r.from_peer == peer.id
            ]
            rel_str = "; ".join(rels) if rels else "none recorded"
            return f"{peer.name} ({peer.id}): {peer.persona.strip()} Relationships: {rel_str}."
        self._record(name, args, [])
        return f"Unknown tool: {name}"

    def _record(self, name: str, args: dict, steps: list[int]) -> None:
        self.log.append(ToolCall(task_id=self.task_id, tool=name, args=args, returned_event_steps=steps))

    @staticmethod
    def _fmt(events) -> str:
        if not events:
            return "No matching messages."
        return "\n".join(f"[step {e.step}] {e.speaker}: {e.content}" for e in events)

    @staticmethod
    def _fmt_hits(hits: list[Retrieved]) -> str:
        if not hits:
            return "No matching messages."
        lines = []
        for h in hits:
            step = h.step if h.step is not None else "?"
            who = f"{h.speaker}: " if h.speaker else ""
            lines.append(f"[step {step}] {who}{h.content}")
        return "\n".join(lines)
