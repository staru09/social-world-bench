"""Eval runner: run the task suite against a run's trajectory, write results.

Reads tasks.json + state.db from a run directory, drives the agent on each task, and
writes results.json (answers + per-task tool calls) and query_log.jsonl (flat log of
every tool call across the suite).
"""

from __future__ import annotations

import json
from pathlib import Path

from swb.eval.agent import run_agent
from swb.eval.tools import ToolBox
from swb.memory import MemoryConfigError, get_backend
from swb.schema import Task, TaskResult
from swb.world.state import WorldState


def _build_backend(memory: str, run_dir: Path, state: WorldState):
    """sqlite reads state.db directly; other backends reconnect via their ingest handle."""
    if memory == "sqlite":
        return get_backend("sqlite")(state)
    handle_path = run_dir / "memory" / f"{memory}.json"
    if not handle_path.exists():
        raise MemoryConfigError(
            f"no ingest handle at {handle_path}; run `swb ingest --run {run_dir} --memory {memory}` first"
        )
    handle = json.loads(handle_path.read_text())
    return get_backend(memory).from_handle(handle, run_dir)


def run_eval(
    run_dir: str | Path, model: str, temperature: float = 0.0, memory: str = "sqlite"
) -> list[TaskResult]:
    run_dir = Path(run_dir)
    tasks = [Task.model_validate(d) for d in json.loads((run_dir / "tasks.json").read_text())]
    state = WorldState(run_dir / "state.db")
    backend = _build_backend(memory, run_dir, state)

    results: list[TaskResult] = []
    query_log: list[dict] = []
    try:
        for task in tasks:
            toolbox = ToolBox(state, task.task_id, backend)
            answer, log = run_agent(task.query, toolbox, model=model, temperature=temperature)
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    type=task.type,
                    query=task.query,
                    expected=task.expected,
                    answer=answer,
                    tool_calls=log,
                )
            )
            query_log.extend(tc.model_dump() for tc in log)
    finally:
        state.close()

    (run_dir / "results.json").write_text(
        json.dumps([r.model_dump() for r in results], indent=2)
    )
    with open(run_dir / "query_log.jsonl", "w") as fh:
        for row in query_log:
            fh.write(json.dumps(row) + "\n")
    return results
