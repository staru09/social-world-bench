"""Post-rollout ledger audit.

Plant-time verification guarantees each gold fact *landed*. This audit checks the
inverse — that nothing planted the fact by accident somewhere else. Two failure modes
it catches, both of which silently corrupt the answer key:

  1. leaked_answer  — a gold answer appears in a NON-gold event. An LLM distractor
                      turn that spontaneously mentions "Lisbon" gives a second, earlier
                      source for a recall answer and breaks temporal ordering.
  2. negation_leak  — the vocabulary a negation task asserts is ABSENT ("dog",
                      "wedding", "tattoo") actually appears. The expected answer "no"
                      would then be wrong.

Scripted mode can't trip either (fixed templates). It matters once --mode llm is on.
Runs that fail the audit should be repaired or discarded, never evaluated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from swb.schema import Event, GoldFact

# Terms the negation tasks assert are absent from the history.
NEGATION_VOCAB = ["dog", "wedding", "tattoo"]


@dataclass
class AuditReport:
    leaked_answers: list[tuple[int, str, str]] = field(default_factory=list)  # (step, fact_id, content)
    negation_leaks: list[tuple[int, str, str]] = field(default_factory=list)  # (step, term, content)
    missing_plants: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.leaked_answers or self.negation_leaks or self.missing_plants)

    def summary(self) -> str:
        if self.ok:
            return "ledger audit: OK (no answer leaks, no negation leaks, all facts planted)"
        lines = ["ledger audit: FAILED"]
        for step, fid, content in self.leaked_answers:
            lines.append(f"  leaked answer for {fid} in non-gold event at step {step}: {content!r}")
        for step, term, content in self.negation_leaks:
            lines.append(f"  negation term {term!r} present at step {step}: {content!r}")
        for fid in self.missing_plants:
            lines.append(f"  gold fact {fid} has no corresponding event")
        return "\n".join(lines)


def _mentions(text: str, term: str) -> bool:
    """Word-boundary match so 'dog' doesn't fire on 'dogged' or 'hotdog'."""
    return re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE) is not None


def audit_ledger(events: list[Event], facts: list[GoldFact]) -> AuditReport:
    report = AuditReport()
    by_step = {e.step: e for e in events}

    for f in facts:
        ev = by_step.get(f.step)
        if ev is None or ev.gold_fact_id != f.fact_id:
            report.missing_plants.append(f.fact_id)
        elif f.answer.lower() not in ev.content.lower():
            report.missing_plants.append(f.fact_id)

    for ev in events:
        if ev.gold_fact_id is not None:
            continue  # gold events are supposed to contain their answer
        for f in facts:
            if f.answer.lower() in ev.content.lower():
                report.leaked_answers.append((ev.step, f.fact_id, ev.content))
        for term in NEGATION_VOCAB:
            if _mentions(ev.content, term):
                report.negation_leaks.append((ev.step, term, ev.content))

    return report
