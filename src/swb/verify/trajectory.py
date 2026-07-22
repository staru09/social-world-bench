"""Trajectory verification: did the agent actually retrieve the right episodes?

This is the second layer, and it's what distinguishes memory use from lucky guessing.
For each task we know the gold facts it requires and the steps those facts live at.
We check whether the agent's logged tool calls ever *returned* those steps.

Two signals per task:
  - trajectory_hit    : did any tool call return an event at a required fact's step?
  - checkpoint_recall : fraction of the task's checkpoint steps the agent's calls touched

Suggestive, not pass/fail (per the design notes): an agent can guess right without
retrieval, and this layer is what exposes that.
"""

from __future__ import annotations

from swb.schema import GoldFact, Task, TaskResult


def _required_steps(task: Task, ledger: dict[str, GoldFact]) -> list[int]:
    steps = []
    for fid in task.required_fact_ids:
        if fid in ledger:
            steps.append(ledger[fid].step)
    return steps


def _returned_steps(result: TaskResult) -> set[int]:
    steps: set[int] = set()
    for tc in result.tool_calls:
        steps.update(tc.returned_event_steps)
    return steps


def trajectory_hit(task: Task, result: TaskResult, ledger: dict[str, GoldFact]) -> bool:
    required = _required_steps(task, ledger)
    if not required:
        # No fact-grounded evidence to check (relationship/negation) — treat as N/A = True.
        return True
    returned = _returned_steps(result)
    return any(s in returned for s in required)


def checkpoint_recall(task: Task, result: TaskResult) -> float:
    if not task.checkpoint_steps:
        return 1.0
    returned = _returned_steps(result)
    hit = sum(1 for s in task.checkpoint_steps if s in returned)
    return hit / len(task.checkpoint_steps)
