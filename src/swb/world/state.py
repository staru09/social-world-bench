from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from swb.schema import Constitution, Event, GoldFact, Peer, Relationship

_SCHEMA = """
CREATE TABLE IF NOT EXISTS world_runs (
    run_id TEXT PRIMARY KEY,
    world_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    num_steps INTEGER NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'in_progress'
);
CREATE TABLE IF NOT EXISTS peers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    persona TEXT NOT NULL,
    traits TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS relationships (
    from_peer TEXT NOT NULL,
    to_peer TEXT NOT NULL,
    type TEXT NOT NULL,
    strength REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    step INTEGER PRIMARY KEY,
    speaker TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT NOT NULL,
    gold_fact_id TEXT
);
CREATE TABLE IF NOT EXISTS gold_facts (
    fact_id TEXT PRIMARY KEY,
    step INTEGER NOT NULL,
    content TEXT NOT NULL,
    answer TEXT NOT NULL,
    subject TEXT NOT NULL,
    kind TEXT NOT NULL,
    peer_refs TEXT NOT NULL,
    checkpoint INTEGER NOT NULL
);
"""


class WorldState:
    """Thin accessor over the per-run SQLite database."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "WorldState":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


    def init_run(self, run_id: str, constitution: Constitution, seed: int, num_steps: int, mode: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO world_runs (run_id, world_id, seed, num_steps, mode, status)"
            " VALUES (?, ?, ?, ?, ?, 'in_progress')",
            (run_id, constitution.world_id, seed, num_steps, mode),
        )
        for p in constitution.peers:
            self.conn.execute(
                "INSERT OR REPLACE INTO peers (id, name, persona, traits) VALUES (?, ?, ?, ?)",
                (p.id, p.name, p.persona, json.dumps(p.traits)),
            )
        for r in constitution.relationships:
            self.conn.execute(
                "INSERT INTO relationships (from_peer, to_peer, type, strength) VALUES (?, ?, ?, ?)",
                (r.from_peer, r.to_peer, r.type, r.strength),
            )
        self.conn.commit()

    def mark_complete(self, run_id: str) -> None:
        self.conn.execute("UPDATE world_runs SET status = 'complete' WHERE run_id = ?", (run_id,))
        self.conn.commit()

    def get_run_id(self) -> str:
        row = self.conn.execute("SELECT run_id FROM world_runs LIMIT 1").fetchone()
        if row is None:
            raise ValueError("no run recorded in this state.db")
        return row["run_id"]


    def append_event(self, event: Event) -> None:
        self.conn.execute(
            "INSERT INTO events (step, speaker, content, tags, gold_fact_id) VALUES (?, ?, ?, ?, ?)",
            (event.step, event.speaker, event.content, json.dumps(event.tags), event.gold_fact_id),
        )
        self.conn.commit()

    def record_gold_fact(self, fact: GoldFact) -> None:
        self.conn.execute(
            "INSERT INTO gold_facts (fact_id, step, content, answer, subject, kind, peer_refs, checkpoint)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact.fact_id,
                fact.step,
                fact.content,
                fact.answer,
                fact.subject,
                fact.kind,
                json.dumps(fact.peer_refs),
                int(fact.checkpoint),
            ),
        )
        self.conn.commit()


    def get_peer(self, peer_id: str) -> Peer | None:
        row = self.conn.execute("SELECT * FROM peers WHERE id = ?", (peer_id,)).fetchone()
        if row is None:
            return None
        return Peer(id=row["id"], name=row["name"], persona=row["persona"], traits=json.loads(row["traits"]))

    def get_relationships(self) -> list[Relationship]:
        rows = self.conn.execute("SELECT * FROM relationships").fetchall()
        return [
            Relationship(from_peer=r["from_peer"], to_peer=r["to_peer"], type=r["type"], strength=r["strength"])
            for r in rows
        ]

    def search_events(self, query: str, limit: int = 10) -> list[Event]:
        """Keyword search over event content (case-insensitive substring)."""
        rows = self.conn.execute(
            "SELECT * FROM events WHERE LOWER(content) LIKE ? ORDER BY step LIMIT ?",
            (f"%{query.lower()}%", limit),
        ).fetchall()
        return [_row_to_event(r) for r in rows]

    def get_events_range(self, start: int, end: int) -> list[Event]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE step >= ? AND step <= ? ORDER BY step", (start, end)
        ).fetchall()
        return [_row_to_event(r) for r in rows]

    def all_events(self) -> list[Event]:
        rows = self.conn.execute("SELECT * FROM events ORDER BY step").fetchall()
        return [_row_to_event(r) for r in rows]

    def all_gold_facts(self) -> list[GoldFact]:
        rows = self.conn.execute("SELECT * FROM gold_facts ORDER BY step").fetchall()
        return [_row_to_gold(r) for r in rows]


def _row_to_event(r: sqlite3.Row) -> Event:
    return Event(
        step=r["step"],
        speaker=r["speaker"],
        content=r["content"],
        tags=json.loads(r["tags"]),
        gold_fact_id=r["gold_fact_id"],
    )


def _row_to_gold(r: sqlite3.Row) -> GoldFact:
    return GoldFact(
        fact_id=r["fact_id"],
        step=r["step"],
        content=r["content"],
        answer=r["answer"],
        subject=r["subject"],
        kind=r["kind"],
        peer_refs=json.loads(r["peer_refs"]),
        checkpoint=bool(r["checkpoint"]),
    )
