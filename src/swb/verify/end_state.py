"""End-state verification: did the agent's answer match the ground truth?

Default is a normalized containment match (the expected string appears in the
answer, or vice versa) which is robust to the agent adding filler words. An optional
LLM judge handles paraphrase-heavy answers, locked to the gold expected string.
"""

from __future__ import annotations

import re


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def end_state_pass(expected: str, answer: str) -> bool:
    """True if the answer conveys the expected fact by normalized containment."""
    exp, ans = _normalize(expected), _normalize(answer)
    if not ans:
        return False
    if exp == ans:
        return True
    # yes/no questions: require an exact normalized token match to avoid "no" matching
    # inside "no problem, yes".
    if exp in {"yes", "no"}:
        return ans.split()[0] == exp if ans else False
    return exp in ans or ans in exp


def end_state_pass_fuzzy(expected: str, answer: str, query: str, model: str) -> bool:
    """LLM-judge fallback, locked to the gold expected string. Requires an API key."""
    from swb.llm import chat

    prompt = (
        "You are grading a memory benchmark answer. The ground-truth answer is fixed; "
        "judge only whether the candidate conveys the same fact.\n\n"
        f"Question: {query}\n"
        f"Ground truth: {expected}\n"
        f"Candidate answer: {answer}\n\n"
        "Reply with exactly one word: PASS or FAIL."
    )
    msg = chat([{"role": "user", "content": prompt}], model=model, temperature=0.0)
    return "pass" in (msg.content or "").strip().lower()
