"""Held-out task generation.

ANTI-LEAKAGE RULE: this module reads only the gold ledger and the constitution —
never the raw trajectory. That is the firewall between "what happened" and "what we
ask": tasks are grounded in facts we planted by construction, so every task has an
unambiguous ground-truth answer without inspecting the noisy transcript.

Deterministic templates (no LLM) for v0. Task types:
  - fact_recall   : ask for a planted fact's answer
  - temporal      : which of two facts happened first (by plant step)
  - relationship  : query the constitution's relationship graph
  - negation      : ask about a fact kind never planted -> expected "no"
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from swb.schema import Constitution, GoldFact, Task

# Fact kinds the world could plausibly contain but that we deliberately never plant,
# used to build negation tasks with a known-"no" answer.
_NEGATION_KINDS = [
    ("dog", "Did anyone mention adopting a dog?"),
    ("wedding", "Did anyone announce a wedding?"),
    ("new tattoo", "Did anyone mention getting a new tattoo?"),
]


def generate_tasks(gold_ledger_path: str | Path, constitution: Constitution) -> list[Task]:
    facts = _load_ledger(gold_ledger_path)
    by_id = {f.fact_id: f for f in facts}
    tasks: list[Task] = []

    # fact_recall — one per planted fact
    for f in facts:
        name = _peer_name(constitution, f.subject)
        tasks.append(
            Task(
                task_id=f"recall_{f.fact_id}",
                type="fact_recall",
                query=f"What {_query_noun(f)} did {name} mention?",
                expected=f.answer,
                required_fact_ids=[f.fact_id],
                checkpoint_steps=[f.step] if f.checkpoint else [],
            )
        )

    # temporal — order the two earliest checkpoint facts (stable, deterministic)
    ordered = sorted(facts, key=lambda x: x.step)
    if len(ordered) >= 2:
        a, b = ordered[0], ordered[1]
        na, nb = _peer_name(constitution, a.subject), _peer_name(constitution, b.subject)
        tasks.append(
            Task(
                task_id="temporal_first",
                type="temporal",
                query=(
                    f"Who shared their news first: {na} ({a.kind}) or {nb} ({b.kind})? "
                    f"Answer with just the name."
                ),
                expected=na,
                required_fact_ids=[a.fact_id, b.fact_id],
                checkpoint_steps=[s for s in (a.step, b.step) if _is_ckpt(by_id, s)],
            )
        )

    # relationship — one per relationship type present in the constitution
    seen_types: set[str] = set()
    for r in constitution.relationships:
        if r.type in seen_types:
            continue
        seen_types.add(r.type)
        frm = _peer_name(constitution, r.from_peer)
        to = _peer_name(constitution, r.to_peer)
        tasks.append(
            Task(
                task_id=f"rel_{r.type}_{r.from_peer}",
                type="relationship",
                query=f"Who is {frm}'s {r.type}? Answer with just the name.",
                expected=to,
                required_fact_ids=[],
                checkpoint_steps=[],
            )
        )

    # negation — ask about kinds never planted
    planted_kinds = {f.kind for f in facts}
    for kind, query in _NEGATION_KINDS:
        if kind not in planted_kinds:
            tasks.append(
                Task(
                    task_id=f"neg_{kind.replace(' ', '_')}",
                    type="negation",
                    query=f"{query} Answer yes or no.",
                    expected="no",
                    required_fact_ids=[],
                    checkpoint_steps=[],
                )
            )

    return tasks


def _load_ledger(path: str | Path) -> list[GoldFact]:
    data = json.loads(Path(path).read_text())
    return [GoldFact.model_validate(d) for d in data]


def _peer_name(constitution: Constitution, peer_id: str) -> str:
    try:
        return constitution.get_peer(peer_id).name
    except KeyError:
        return peer_id


def _is_ckpt(by_id: dict[str, GoldFact], step: int) -> bool:
    return any(f.step == step and f.checkpoint for f in by_id.values())


# The query noun is stored in script.yaml, but the ledger doesn't carry it; derive a
# readable noun from the fact kind so the generator stays ledger-only.
_KIND_NOUN = {
    "pet": "pet",
    "job": "job change",
    "relocation": "city",
    "hobby": "hobby",
    "vehicle": "vehicle",
}


def _query_noun(f: GoldFact) -> str:
    return _KIND_NOUN.get(f.kind, f.kind)


def write_tasks(tasks: list[Task], path: str | Path) -> None:
    payload = [t.model_dump() for t in tasks]
    Path(path).write_text(json.dumps(payload, indent=2))
