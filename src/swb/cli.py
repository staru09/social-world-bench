"""swb command-line interface.

Pipeline stages, each reading only the previous stage's artifacts:
  swb rollout   world setup + rollout  -> runs/<seed>/ (trajectory, ledger, state.db)
  swb gentasks  task generation        -> runs/<seed>/tasks.json
  swb ingest    load trajectory into a memory backend -> runs/<seed>/memory/<backend>.json
  swb eval      agent eval             -> runs/<seed>/results.json, query_log.jsonl
  swb report    verification + report  -> stdout / report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from swb.world.constitution import ConstitutionError, load_constitution, resolve_world


@click.group()
def cli() -> None:
    """Social World Memory Benchmark."""


@cli.command()
@click.option("--world", required=True, help="World name (e.g. generic_social_v1) or path to constitution.yaml")
@click.option("--steps", type=int, default=None, help="Number of rollout steps (default: constitution value)")
@click.option("--seed", type=int, default=None, help="RNG seed (default: constitution value)")
@click.option("--mode", type=click.Choice(["scripted", "llm"]), default="scripted")
@click.option("--out", type=click.Path(), default=None, help="Output dir (default: runs/<seed>/)")
def rollout(world: str, steps: int | None, seed: int | None, mode: str, out: str | None) -> None:
    """Stage 1+2/3: build the world and roll out a history."""
    from swb.llm import LLMConfigError
    from swb.rollout.engine import run_rollout

    try:
        cpath = resolve_world(world)
        constitution = load_constitution(cpath)
    except ConstitutionError as e:
        raise click.ClickException(str(e))
    steps = steps if steps is not None else constitution.rollout.num_steps
    seed = seed if seed is not None else constitution.rollout.rng_seed
    out = out if out is not None else f"runs/{seed}"

    try:
        summary = run_rollout(constitution, cpath, out_dir=out, num_steps=steps, seed=seed, mode=mode)
    except LLMConfigError as e:
        raise click.ClickException(str(e))
    click.echo(json.dumps(summary, indent=2))


@cli.command()
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True))
def gentasks(run_dir: str) -> None:
    """Stage 4: derive held-out tasks from the run's gold ledger."""
    from swb.tasks.generator import generate_tasks, write_tasks

    run_dir = Path(run_dir)
    constitution = load_constitution(run_dir / "constitution.yaml")
    tasks = generate_tasks(run_dir / "gold_ledger.json", constitution)
    write_tasks(tasks, run_dir / "tasks.json")
    click.echo(f"wrote {len(tasks)} tasks to {run_dir / 'tasks.json'}")
    for t in tasks:
        click.echo(f"  [{t.type}] {t.query}  -> {t.expected!r}")


@cli.command()
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True))
@click.option("--memory", type=click.Choice(["mem0", "honcho", "supermemory"]), required=True)
def ingest(run_dir: str, memory: str) -> None:
    """Stage 4.5: ingest the run's trajectory into a memory backend for retrieval."""
    from swb.memory import MemoryConfigError, get_backend
    from swb.world.state import WorldState

    run_dir_p = Path(run_dir)
    state = WorldState(run_dir_p / "state.db")
    try:
        namespace = f"swb-{state.get_run_id()}"
        backend = get_backend(memory)(namespace)
        handle = backend.ingest(state.all_events())
    except MemoryConfigError as e:
        raise click.ClickException(str(e))
    finally:
        state.close()

    mem_dir = run_dir_p / "memory"
    mem_dir.mkdir(exist_ok=True)
    handle_path = mem_dir / f"{memory}.json"
    handle_path.write_text(json.dumps(handle, indent=2))
    click.echo(
        f"ingested {handle['n_events']} events into {memory} "
        f"(namespace={handle['namespace']}) -> {handle_path}"
    )


@cli.command()
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True))
@click.option("--model", required=True, help="OpenRouter model, e.g. openai/gpt-4o-mini")
@click.option("--temperature", type=float, default=0.0)
@click.option(
    "--memory",
    type=click.Choice(["sqlite", "mem0", "honcho", "supermemory"]),
    default="sqlite",
    help="Retrieval backend for search_events (default: sqlite baseline)",
)
def eval(run_dir: str, model: str, temperature: float, memory: str) -> None:
    """Stage 5: run the eval agent over the task suite."""
    from swb.eval.runner import run_eval
    from swb.llm import LLMConfigError
    from swb.memory import MemoryConfigError

    try:
        results = run_eval(run_dir, model=model, temperature=temperature, memory=memory)
    except (LLMConfigError, MemoryConfigError) as e:
        raise click.ClickException(str(e))
    click.echo(
        f"evaluated {len(results)} tasks with memory={memory} -> {Path(run_dir) / 'results.json'}"
    )


@cli.command()
@click.option("--runs", "run_dirs", required=True, multiple=True, type=click.Path(exists=True))
@click.option("--out", type=click.Path(), default=None, help="Write report JSON here")
def report(run_dirs: tuple[str, ...], out: str | None) -> None:
    """Stage 6: two-layer verification + multi-seed aggregation."""
    from swb.verify.report import aggregate

    rep = aggregate(list(run_dirs))
    text = json.dumps(rep, indent=2)
    click.echo(text)
    if out:
        Path(out).write_text(text)


if __name__ == "__main__":
    cli()
