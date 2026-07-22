from __future__ import annotations

from swb.schema import Constitution, Event, GoldFact
from swb.world.state import WorldState


class ConstitutionViolation(ValueError):
    """Raised when a proposed step breaks the world's interaction rules."""


def apply_step(
    state: WorldState,
    constitution: Constitution,
    step: int,
    speaker: str,
    content: str,
    tags: list[str] | None = None,
    gold_fact: GoldFact | None = None,
) -> Event:
    """Validate a proposed turn against the constitution, then commit it.

    If `gold_fact` is provided, the event carries its id and the fact is recorded
    in the same transaction sequence — this is the plant-time write that makes the
    ledger trustworthy.
    """
    tags = list(tags or [])

    if speaker not in constitution.peer_ids():
        raise ConstitutionViolation(f"unknown speaker: {speaker}")
    allowed = constitution.rules.allowed_speakers
    if allowed is not None and speaker not in allowed:
        raise ConstitutionViolation(f"speaker {speaker} not permitted to speak")

    if gold_fact is not None:
        # Guard: in LLM mode the utterance is generated; only record the fact if it
        # actually landed in the content. Scripted mode always satisfies this.
        if gold_fact.answer.lower() not in content.lower():
            raise ConstitutionViolation(
                f"gold fact {gold_fact.fact_id} not present in emitted content; refusing to record ledger entry"
            )

    event = Event(
        step=step,
        speaker=speaker,
        content=content,
        tags=tags,
        gold_fact_id=gold_fact.fact_id if gold_fact else None,
    )
    state.append_event(event)
    if gold_fact is not None:
        state.record_gold_fact(gold_fact)
    return event
