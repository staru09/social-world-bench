# Social World Memory Benchmark (`swb`)

A benchmark for evaluating **agent memory systems**. We generate a synthetic
multi-peer social world, roll out a long conversation history with a handful of
planted "gold facts" buried in noise, then test whether an agent can answer
held-out questions that require recalling those facts — verifying both the final
answer and whether the agent actually retrieved the right episodes.

## Why

An agent's memory system has to surface a few relevant signals out of a history far
too long to fit in context. This benchmark makes the signals **known by
construction**: we decide the facts before the rollout, plant them at chosen
timesteps, and record them in a *gold ledger*. Because both the tasks and the
answer key come from that ledger, verification needs no fragile transcript parsing.

## Pipeline

```
world setup ──▶ rollout ──▶ task gen ──▶ [ingest] ──▶ eval ──▶ verify/report
constitution   trajectory    tasks.json   memory      agent    two-layer scoring
+ schedule     + gold ledger              backend     + tools   (end state + trajectory)
```

Each stage reads only the previous stage's artifacts. In particular the task
generator reads **only the gold ledger** (never the raw transcript) — the
anti-leakage firewall between "what happened" and "what we ask."

`ingest` is optional: it loads the trajectory into a real memory service (mem0,
honcho, supermemory) so `eval` can retrieve through *that system's own retriever*.
Skip it and `eval` falls back to the built-in `sqlite` baseline. See
[Memory backends](#memory-backends).

### Two-layer verification

- **End state** — did the agent's answer match the gold fact? (normalized match, optional LLM judge)
- **Trajectory** — did the agent's logged tool calls actually return the events where the required facts were planted? This is what distinguishes memory *use* from a lucky guess.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```

Memory backends are optional extras — install only the ones you'll use:

```bash
uv pip install -e ".[mem0]"          # or .[honcho] / .[supermemory]
uv pip install -e ".[dev,mem0,honcho,supermemory]"   # everything
```

LLM stages (`--mode llm`, `swb eval`) speak the OpenAI wire format and work with
**OpenAI or OpenRouter** — set whichever key you have (OpenAI wins if both are set;
pin explicitly with `SWB_PROVIDER`):

```bash
export OPENAI_API_KEY=...        # or OPENROUTER_API_KEY=...
```

Memory backends each need their own key, read only when that backend is used:

```bash
export MEM0_API_KEY=...           # --memory mem0 (hosted)
export HONCHO_API_KEY=...         # --memory honcho
export SUPERMEMORY_API_KEY=...    # --memory supermemory
```

Scripted rollout, task generation, and the full test suite need no API key.

## Usage

```bash
# 1+2. Build the world and roll out a deterministic history
swb rollout --world generic_social_v1 --steps 100 --seed 42 --out runs/42

# 3. (optional) LLM-simulated peers instead of scripted templates
swb rollout --world generic_social_v1 --steps 100 --seed 42 --mode llm --out runs/42

# 4. Generate held-out tasks from the gold ledger
swb gentasks --run runs/42

# 5. Evaluate an agent — the default `sqlite` baseline needs no memory service
swb eval --run runs/42 --model openai/gpt-4o-mini

# 6. Two-layer verification + multi-seed report
swb report --runs runs/42 --runs runs/43 --out report.json
```

To evaluate against a real memory system instead of the baseline, insert an
`ingest` step between 4 and 5 — see below.

## Memory backends

`swb eval` retrieves through a pluggable backend selected with `--memory`:

| Backend | Retrieval | Needs |
|---------|-----------|-------|
| `sqlite` *(default)* | substring `LIKE` over `state.db` — the baseline | nothing |
| `mem0` | hosted mem0 semantic search | `MEM0_API_KEY`, `.[mem0]` |
| `honcho` | honcho hybrid (full-text + semantic) | `HONCHO_API_KEY`, `.[honcho]` |
| `supermemory` | supermemory semantic search | `SUPERMEMORY_API_KEY`, `.[supermemory]` |

The non-baseline backends need a one-time **ingest** that loads the whole trajectory
into the service (one item per event, tagged with its step). Ingest once, then eval
as many times as you like against the persisted store:

```bash
# Load runs/42's trajectory into mem0 (writes runs/42/memory/mem0.json)
swb ingest --run runs/42 --memory mem0

# Evaluate through mem0's retriever, then report
swb eval   --run runs/42 --memory mem0 --model openai/gpt-4o-mini
swb report --runs runs/42
```

Swap `mem0` for `honcho` or `supermemory` to benchmark another system. How it works:

- **Ingest** attaches `{"step": N}` metadata to every event. **Search** reads that
  step back off each hit, so *trajectory* verification (did the agent surface the
  event where the fact was planted?) works identically across all backends.
- `eval` writes `results.json` last-run-wins, so run `swb report` after each
  backend's eval to capture its numbers before the next overwrites them.
- Compared to the `sqlite` baseline, a working memory backend should lift
  `fact_recall` pass rate and, crucially, its `trajectory_hit_rate` above zero —
  evidence the agent is *retrieving* the planted episodes, not guessing.

## Run artifacts

| File | Stage | Contents |
|------|-------|----------|
| `state.db` | rollout | SQLite: seed state + full trajectory + gold ledger |
| `trajectory.jsonl` | rollout | every event, in order (the haystack) |
| `gold_ledger.json` | rollout | only planted facts (the answer key) |
| `constitution.yaml` | rollout | snapshot of the world definition used |
| `tasks.json` | gentasks | held-out questions + expected answers |
| `memory/<backend>.json` | ingest | handle for a memory backend (namespace + event count) |
| `results.json` | eval | agent answers + per-task tool calls |
| `query_log.jsonl` | eval | flat log of every tool call (for trajectory verification) |

## Layout

```
worlds/generic_social_v1/   constitution.yaml (peers, rules) + script.yaml (fact schedule)
src/swb/schema.py           Pydantic models shared across stages
src/swb/world/              constitution loader, SQLite state, transition function
src/swb/rollout/            scripted + LLM rollout engines
src/swb/tasks/              held-out task generator (ledger-only)
src/swb/memory/             pluggable retrieval backends (sqlite, mem0, honcho, supermemory)
src/swb/eval/               search tools, agent loop, runner
src/swb/verify/             end-state + trajectory checks, report aggregation
cli.py                      swb rollout | gentasks | ingest | eval | report
```