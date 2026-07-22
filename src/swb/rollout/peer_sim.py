"""LLM peer simulator.

The engine still owns the golden truth: at each scheduled fact step it *decides* the
fact and instructs the speaking peer to mention it. The LLM only controls phrasing.
The transition function then verifies the emitted utterance actually contains the
fact's answer before the ledger entry is recorded (with a bounded retry here), so a
ledger entry only exists if the fact demonstrably landed in the history.

Non-fact turns are split by the constitution's `noise_ratio`:
  - filler      (noise_ratio)     : pure small talk, trivially irrelevant
  - distractor  (1 - noise_ratio) : personal-sounding news that is NOT a gold fact —
                                    hard negatives that stop retrieval from succeeding
                                    just by finding "the one substantive message"
Distractors are explicitly steered away from the gold fact topics and from the
negation vocabulary, and `verify.ledger_audit` checks after the fact that none leaked.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterator, Optional

import yaml

from swb.llm import chat
from swb.schema import Constitution, GoldFact

_MAX_FACT_RETRIES = 3

# Topics distractors must avoid so they cannot accidentally answer a negation task.
FORBIDDEN_DISTRACTOR_TOPICS = ["dog", "wedding", "engaged", "tattoo"]


def _persona_system(constitution: Constitution, speaker: str) -> str:
    p = constitution.get_peer(speaker)
    return (
        f"You are {p.name}. {p.persona.strip()} "
        "You are chatting casually in a group thread with friends. "
        "Reply with a single short, natural message (one or two sentences). "
        "Write only the message text: do not prefix it with your name, do not use "
        "quotation marks, and do not narrate."
    )


def _recent_window(history: list[tuple[str, str]], k: int = 6) -> list[dict]:
    """Feed recent turns back as context, attributed by speaker."""
    return [{"role": "user", "content": f"{spk}: {content}"} for spk, content in history[-k:]]


def _clean(content: str, speaker: str, constitution: Constitution) -> str:
    """Strip a leading 'alex:' / 'Alex:' speaker label the model may have echoed.

    The recent-window context is formatted as '<speaker>: <text>', and models tend to
    mimic that shape in their own reply. Left in, it pollutes the trajectory.
    """
    content = content.strip()
    try:
        name = constitution.get_peer(speaker).name
    except KeyError:
        name = speaker
    pattern = rf"^\s*(?:{re.escape(speaker)}|{re.escape(name)})\s*:\s*"
    prev = None
    while prev != content:  # handle a doubled 'alex: alex:' echo
        prev = content
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)
    return content.strip()


def llm_turns(
    constitution: Constitution,
    script_path: str | Path,
    num_steps: int,
    seed: int,
) -> Iterator[tuple[str, str, Optional[GoldFact]]]:
    """Yield (speaker, content, gold_fact|None), generating content with the LLM."""
    data = yaml.safe_load(Path(script_path).read_text())
    facts = {f["step"]: f for f in data.get("facts", []) if f["step"] < num_steps}
    distractor_topics = data.get("distractor_topics", [])

    rng = random.Random(seed)
    peer_ids = sorted(constitution.peer_ids())
    speakable = constitution.rules.allowed_speakers or peer_ids
    model = constitution.rollout.llm_model
    temperature = constitution.rollout.llm_temperature
    noise_ratio = constitution.rules.noise_ratio

    # Gold answers must never surface outside their planted step.
    gold_answers = [f["answer"] for f in facts.values()]
    banned = gold_answers + FORBIDDEN_DISTRACTOR_TOPICS

    history: list[tuple[str, str]] = []

    for step in range(num_steps):
        if step in facts:
            f = facts[step]
            speaker = f["subject"] if f["subject"] in speakable else rng.choice(speakable)
            gold = GoldFact(
                fact_id=f["fact_id"],
                step=step,
                content=f["content"].format(answer=f["answer"]),
                answer=f["answer"],
                subject=f["subject"],
                kind=f["kind"],
                peer_refs=[f["subject"]],
                checkpoint=bool(f.get("checkpoint", False)),
            )
            system = _persona_system(constitution, speaker) + (
                " In this message you MUST naturally share this personal news, "
                f'including the exact phrase "{f["answer"]}": {f["content"].format(answer=f["answer"])}'
            )
            content = _generate_with_fact(
                system, history, f["answer"], model, temperature, speaker, constitution
            )
            history.append((speaker, content))
            yield speaker, content, gold
        else:
            speaker = rng.choice(speakable)
            is_filler = rng.random() < noise_ratio
            if is_filler:
                instruction = (
                    " Make small talk about everyday trivia (food, weather, commute, TV). "
                    "Share no personal news of any kind."
                )
            else:
                topic = rng.choice(distractor_topics) if distractor_topics else "a minor work update"
                instruction = (
                    f" Chat about this minor, everyday topic: {topic}. "
                    "It should sound mildly personal but be unremarkable — "
                    "NOT major life news."
                )
            instruction += " Never mention: " + ", ".join(banned) + "."
            system = _persona_system(constitution, speaker) + instruction

            msgs = [{"role": "system", "content": system}] + _recent_window(history)
            msgs.append({"role": "user", "content": "Write your next short message."})
            content = _clean(
                chat(msgs, model=model, temperature=temperature).content, speaker, constitution
            )
            history.append((speaker, content))
            yield speaker, content, None


def _generate_with_fact(
    system: str,
    history: list[tuple[str, str]],
    answer: str,
    model: str,
    temperature: float,
    speaker: str,
    constitution: Constitution,
) -> str:
    """Generate until the utterance contains the answer verbatim, or fall back."""
    for _ in range(_MAX_FACT_RETRIES):
        msgs = [{"role": "system", "content": system}] + _recent_window(history)
        msgs.append({"role": "user", "content": "Write your next short message."})
        content = _clean(chat(msgs, model=model, temperature=temperature).content, speaker, constitution)
        if answer.lower() in content.lower():
            return content
    # Deterministic fallback guarantees the fact lands so the ledger stays valid.
    return f"By the way — {answer}."
