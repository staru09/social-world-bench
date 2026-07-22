"""Deterministic scripted rollout.

Given a fact schedule and noise templates, produce a fixed sequence of (speaker,
content, gold_fact) turns for a seed. Same seed -> identical sequence. This is the
fast, no-LLM path used for tests and the demo artifacts, and it defines the golden
truth by construction: facts are placed at their scheduled steps and every other
turn is noise carrying no ledger entry.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Optional

import yaml

from swb.schema import Constitution, GoldFact


class ScriptedPlan:
    """Loads a world's script.yaml and yields turns for a rollout."""

    def __init__(self, script_path: str | Path):
        data = yaml.safe_load(Path(script_path).read_text())
        self.facts = data.get("facts", [])
        self.noise_templates = data.get("noise_templates", [])
        self.distractor_templates = data.get("distractor_templates", [])
        if not self.noise_templates:
            raise ValueError("script.yaml must define at least one noise_template")

    def turns(
        self, constitution: Constitution, num_steps: int, seed: int
    ) -> Iterator[tuple[str, str, Optional[GoldFact]]]:
        """Yield (speaker, content, gold_fact|None) for steps 0..num_steps-1.

        Fact-planting steps are fixed by the schedule. Noise turns use the seeded
        RNG for speaker and template choice, so the whole sequence is reproducible.
        """
        rng = random.Random(seed)
        peer_ids = sorted(constitution.peer_ids())
        speakable = constitution.rules.allowed_speakers or peer_ids

        fact_by_step: dict[int, dict] = {}
        for f in self.facts:
            if f["step"] < num_steps:
                fact_by_step[f["step"]] = f

        for step in range(num_steps):
            if step in fact_by_step:
                f = fact_by_step[step]
                content = f["content"].format(answer=f["answer"])
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
                # The subject speaks their own news; fall back to any speaker if the
                # subject is not permitted to speak.
                speaker = f["subject"] if f["subject"] in speakable else rng.choice(speakable)
                yield speaker, content, gold
            else:
                speaker = rng.choice(speakable)
                # noise_ratio splits non-fact turns into pure filler vs distractors
                # (personal-sounding hard negatives that carry no gold fact).
                if rng.random() < constitution.rules.noise_ratio or not self.distractor_templates:
                    content = rng.choice(self.noise_templates)
                else:
                    content = rng.choice(self.distractor_templates)
                yield speaker, content, None
