"""Memory backend tests — no network.

The cloud SDKs (mem0/honcho/supermemory) are never imported live: each test injects a
fake module into sys.modules, so we verify our ingest metadata and our search->Retrieved
mapping without any API key or network call.
"""

from __future__ import annotations

import sys
import types

import pytest

from swb.eval.tools import ToolBox
from swb.memory import MemoryConfigError, get_backend
from swb.memory.base import MemoryBackend, Retrieved
from swb.memory.sqlite_backend import SqliteBackend
from swb.schema import Event, GoldFact, Task, TaskResult
from swb.verify.trajectory import trajectory_hit
from swb.world.state import WorldState


@pytest.fixture(autouse=True)
def clear_keys(monkeypatch):
    for var in ("MEM0_API_KEY", "HONCHO_API_KEY", "SUPERMEMORY_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# --- the invariant that makes backends interchangeable -----------------------------------


class _FakeBackend(MemoryBackend):
    name = "fake"

    def __init__(self, hits):
        self._hits = hits

    @classmethod
    def from_handle(cls, handle, run_dir=None):
        return cls([])

    def ingest(self, events):
        return {"backend": self.name, "namespace": "x", "n_events": len(events)}

    def search(self, query, limit=10):
        return self._hits


def test_backend_step_flows_into_trajectory_verification():
    # A hit carrying step=5 must surface as a returned_event_step, which is what
    # trajectory verification scores on.
    backend = _FakeBackend([Retrieved(content="Alex adopted a cat", step=5, speaker="alex")])
    box = ToolBox(state=None, task_id="t1", backend=backend)
    box.dispatch("search_events", {"query": "pet"})

    assert box.log[0].returned_event_steps == [5]

    task = Task(task_id="t1", type="fact_recall", query="q", expected="a cat", required_fact_ids=["f_pet"])
    result = TaskResult(task_id="t1", type="fact_recall", query="q", expected="a cat", answer="a cat", tool_calls=box.log)
    ledger = {"f_pet": GoldFact(fact_id="f_pet", step=5, content="c", answer="a cat", subject="alex", kind="pet")}
    assert trajectory_hit(task, result, ledger) is True


def test_unrecoverable_step_is_excluded_from_trajectory():
    backend = _FakeBackend([Retrieved(content="a fused memory", step=None)])
    box = ToolBox(state=None, task_id="t1", backend=backend)
    box.dispatch("search_events", {"query": "pet"})
    assert box.log[0].returned_event_steps == []


# --- sqlite baseline backend -------------------------------------------------------------


def test_sqlite_backend_maps_events_with_steps(tmp_path):
    state = WorldState(tmp_path / "s.db")
    state.append_event(Event(step=3, speaker="alex", content="I adopted a cat named Miso"))
    state.append_event(Event(step=7, speaker="sam", content="the weather is nice"))

    backend = SqliteBackend(state)
    hits = backend.search("cat")
    assert [(h.step, h.speaker) for h in hits] == [(3, "alex")]
    assert backend.ingest(state.all_events()) == {"backend": "sqlite", "namespace": None, "n_events": 2}


# --- registry / config errors ------------------------------------------------------------


def test_unknown_backend_raises():
    with pytest.raises(MemoryConfigError, match="unknown memory backend"):
        get_backend("redis")


def test_missing_key_raises_before_network():
    # No MEM0_API_KEY set -> construction fails cleanly, never touches the SDK/network.
    with pytest.raises(MemoryConfigError, match="MEM0_API_KEY"):
        get_backend("mem0")("swb-run")


# --- SDK mapping (fake modules injected) -------------------------------------------------


def _install_fake_module(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


def test_mem0_ingest_attaches_step_and_search_maps(monkeypatch):
    calls = {"add": [], "search": []}

    class FakeMemoryClient:
        def add(self, messages, **kwargs):
            calls["add"].append((messages, kwargs))

        def search(self, query, **kwargs):
            calls["search"].append((query, kwargs))
            return {"results": [
                {"memory": "alex: I adopted a cat named Miso",
                 "score": 0.9,
                 "metadata": {"step": 5, "speaker": "alex"}},
            ]}

    fake = types.ModuleType("mem0")
    fake.MemoryClient = FakeMemoryClient
    _install_fake_module(monkeypatch, "mem0", fake)
    monkeypatch.setenv("MEM0_API_KEY", "x")

    backend = get_backend("mem0")("swb-run")
    handle = backend.ingest([Event(step=5, speaker="alex", content="I adopted a cat named Miso")])
    assert handle == {"backend": "mem0", "namespace": "swb-run", "n_events": 1}
    # metadata carries the step, and infer=False keeps the turn raw
    _, add_kwargs = calls["add"][0]
    assert add_kwargs["metadata"] == {"step": 5, "speaker": "alex"}
    assert add_kwargs["infer"] is False
    assert add_kwargs["run_id"] == "swb-run"

    hits = backend.search("pet", limit=3)
    assert hits[0].step == 5 and hits[0].speaker == "alex" and hits[0].score == 0.9
    # scoped to this run's namespace
    assert calls["search"][0][1]["filters"] == {"AND": [{"run_id": "swb-run"}]}


def test_honcho_ingest_and_search_maps(monkeypatch):
    added = {"peers": [], "messages": []}

    class FakeMessage:
        def __init__(self, peer_id, content, metadata):
            self.peer_id, self.content, self.metadata = peer_id, content, metadata

    class FakePeer:
        def __init__(self, pid):
            self.id = pid

        def message(self, content, metadata=None):
            return FakeMessage(self.id, content, metadata or {})

    class FakeSession:
        def add_peers(self, peers):
            added["peers"].extend(p.id for p in peers)

        def add_messages(self, msgs):
            added["messages"].extend(msgs)

        def search(self, query):
            return [FakeMessage("alex", "I adopted a cat named Miso", {"step": 5})]

    class FakeHoncho:
        def __init__(self, **kwargs):
            pass

        def peer(self, pid):
            return FakePeer(pid)

        def session(self, sid):
            return FakeSession()

    fake = types.ModuleType("honcho")
    fake.Honcho = FakeHoncho
    _install_fake_module(monkeypatch, "honcho", fake)
    monkeypatch.setenv("HONCHO_API_KEY", "x")

    backend = get_backend("honcho")("swb-run")
    backend.ingest([
        Event(step=5, speaker="alex", content="I adopted a cat named Miso"),
        Event(step=6, speaker="sam", content="nice"),
    ])
    assert set(added["peers"]) == {"alex", "sam"}
    assert added["messages"][0].metadata == {"step": 5}

    hits = backend.search("pet", limit=5)
    assert hits[0].step == 5 and hits[0].speaker == "alex"


def test_supermemory_ingest_and_search_maps(monkeypatch):
    calls = {"add": []}

    class Chunk:
        def __init__(self, content):
            self.content = content

    class Result:
        def __init__(self):
            self.metadata = {"step": 5, "speaker": "alex"}
            self.chunks = [Chunk("alex: I adopted a cat named Miso")]
            self.score = 0.8

    class FakeSearchResp:
        results = [Result()]

    class FakeDocuments:
        def list_processing(self, **kwargs):
            return types.SimpleNamespace(results=[])  # nothing pending -> ingest returns fast

    class FakeSearch:
        def documents(self, **kwargs):
            return FakeSearchResp()

    class FakeSupermemory:
        def __init__(self, **kwargs):
            self.documents = FakeDocuments()
            self.search = FakeSearch()

        def add(self, **kwargs):
            calls["add"].append(kwargs)

    fake = types.ModuleType("supermemory")
    fake.Supermemory = FakeSupermemory
    _install_fake_module(monkeypatch, "supermemory", fake)
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "x")

    backend = get_backend("supermemory")("swb-run")
    backend.ingest([Event(step=5, speaker="alex", content="I adopted a cat named Miso")])
    assert calls["add"][0]["metadata"] == {"step": 5, "speaker": "alex"}
    assert calls["add"][0]["container_tags"] == ["swb-run"]

    hits = backend.search("pet", limit=3)
    assert hits[0].step == 5 and hits[0].speaker == "alex"
    assert "cat named Miso" in hits[0].content
