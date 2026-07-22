"""Rollout orchestration: run the simulation, write the run artifacts.

Produces, in `out_dir`:
  - state.db           the SQLite trajectory + gold ledger + seed state
  - trajectory.jsonl   every event, in order (the haystack)
  - gold_ledger.json   only the planted facts (the answer key)
  - constitution.yaml  a snapshot of the world definition used
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from swb.rollout.scripted import ScriptedPlan
from swb.schema import Constitution
from swb.verify.ledger_audit import audit_ledger
from swb.world.state import WorldState
from swb.world.transition import apply_step


def run_rollout(
    constitution: Constitution,
    constitution_path: str | Path,
    out_dir: str | Path,
    num_steps: int,
    seed: int,
    mode: str = "scripted",
) -> dict:
    """Execute a rollout and write artifacts. Returns a small summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "state.db"
    if db_path.exists():
        db_path.unlink()

    run_id = f"{constitution.world_id}-seed{seed}"
    state = WorldState(db_path)
    state.init_run(run_id, constitution, seed=seed, num_steps=num_steps, mode=mode)

    if mode == "scripted":
        script_path = Path(constitution_path).parent / "script.yaml"
        plan = ScriptedPlan(script_path)
        turn_source = plan.turns(constitution, num_steps=num_steps, seed=seed)
    elif mode == "llm":
        from swb.rollout.peer_sim import llm_turns  # lazy import; needs API key

        script_path = Path(constitution_path).parent / "script.yaml"
        turn_source = llm_turns(constitution, script_path, num_steps=num_steps, seed=seed)
    else:
        raise ValueError(f"unknown rollout mode: {mode}")

    n_facts = 0
    for step, (speaker, content, gold) in enumerate(turn_source):
        apply_step(state, constitution, step=step, speaker=speaker, content=content, gold_fact=gold)
        if gold is not None:
            n_facts += 1

    state.mark_complete(run_id)

    # Audit before the run is usable: plant-time checks prove each fact landed; this
    # proves nothing planted one by accident elsewhere (which would corrupt the key).
    audit = audit_ledger(state.all_events(), state.all_gold_facts())

    _write_trajectory(state, out_dir / "trajectory.jsonl")
    _write_gold_ledger(state, out_dir / "gold_ledger.json")
    shutil.copyfile(constitution_path, out_dir / "constitution.yaml")
    (out_dir / "audit.json").write_text(
        json.dumps(
            {
                "ok": audit.ok,
                "leaked_answers": audit.leaked_answers,
                "negation_leaks": audit.negation_leaks,
                "missing_plants": audit.missing_plants,
            },
            indent=2,
        )
    )

    state.close()
    return {
        "run_id": run_id,
        "steps": num_steps,
        "facts": n_facts,
        "audit_ok": audit.ok,
        "audit": audit.summary(),
        "out": str(out_dir),
    }


def _write_trajectory(state: WorldState, path: Path) -> None:
    with open(path, "w") as fh:
        for ev in state.all_events():
            fh.write(ev.model_dump_json() + "\n")


def _write_gold_ledger(state: WorldState, path: Path) -> None:
    facts = [f.model_dump() for f in state.all_gold_facts()]
    path.write_text(json.dumps(facts, indent=2))
