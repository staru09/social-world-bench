"""Scoring + multi-seed aggregation.

Turns a run's tasks.json + results.json + gold_ledger.json into per-task Verdicts,
then aggregates across one or more runs: per-task-type pass rate with a Wilson 95%
confidence interval, mean tool calls, and checkpoint recall distribution.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from swb.schema import GoldFact, Task, TaskResult, Verdict
from swb.verify.end_state import end_state_pass
from swb.verify.trajectory import checkpoint_recall, trajectory_hit


def verify_run(run_dir: str | Path) -> list[Verdict]:
    run_dir = Path(run_dir)
    tasks = {t.task_id: t for t in (Task.model_validate(d) for d in _load(run_dir / "tasks.json"))}
    results = [TaskResult.model_validate(d) for d in _load(run_dir / "results.json")]
    ledger = {f.fact_id: f for f in (GoldFact.model_validate(d) for d in _load(run_dir / "gold_ledger.json"))}

    verdicts: list[Verdict] = []
    for res in results:
        task = tasks[res.task_id]
        verdicts.append(
            Verdict(
                task_id=task.task_id,
                type=task.type,
                end_state_pass=end_state_pass(task.expected, res.answer),
                trajectory_hit=trajectory_hit(task, res, ledger),
                checkpoint_recall=checkpoint_recall(task, res),
            )
        )
    return verdicts


def wilson_ci(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = passes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate(run_dirs: list[str | Path]) -> dict:
    """Aggregate verdicts across runs into a report dict."""
    all_verdicts: list[Verdict] = []
    total_tool_calls = 0
    n_tasks = 0
    for rd in run_dirs:
        rd = Path(rd)
        all_verdicts.extend(verify_run(rd))
        results = [TaskResult.model_validate(d) for d in _load(rd / "results.json")]
        for r in results:
            total_tool_calls += len(r.tool_calls)
            n_tasks += 1

    by_type: dict[str, list[Verdict]] = {}
    for v in all_verdicts:
        by_type.setdefault(v.type, []).append(v)

    per_type = {}
    for t, vs in sorted(by_type.items()):
        passes = sum(1 for v in vs if v.end_state_pass)
        lo, hi = wilson_ci(passes, len(vs))
        traj = sum(1 for v in vs if v.trajectory_hit)
        recall_vals = [v.checkpoint_recall for v in vs]
        per_type[t] = {
            "n": len(vs),
            "pass_rate": round(passes / len(vs), 3),
            "wilson95": [round(lo, 3), round(hi, 3)],
            "trajectory_hit_rate": round(traj / len(vs), 3),
            "mean_checkpoint_recall": round(sum(recall_vals) / len(vs), 3),
        }

    overall_pass = sum(1 for v in all_verdicts if v.end_state_pass)
    return {
        "runs": [str(r) for r in run_dirs],
        "n_tasks": len(all_verdicts),
        "overall_pass_rate": round(overall_pass / len(all_verdicts), 3) if all_verdicts else 0.0,
        "mean_tool_calls_per_task": round(total_tool_calls / n_tasks, 2) if n_tasks else 0.0,
        "per_type": per_type,
    }


def _load(path: Path) -> list:
    return json.loads(Path(path).read_text())
