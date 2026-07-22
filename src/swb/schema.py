from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Peer(BaseModel):
    """A participant in the social world."""

    id: str
    name: str
    persona: str = Field(..., description="2-4 sentence backstory used to prompt the LLM peer simulator")
    traits: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """A directed edge between two peers. Also serves as ground truth for
    relationship-type tasks."""

    from_peer: str
    to_peer: str
    type: str = Field(..., description="e.g. friend, colleague, roommate")
    strength: float = 1.0


class InteractionRules(BaseModel):
    """The constitution's constraints on how peers may interact."""

    allowed_speakers: Optional[list[str]] = Field(
        default=None,
        description="Peer ids allowed to initiate. None means all peers may speak.",
    )
    noise_ratio: float = Field(
        0.6, ge=0.0, le=1.0, description="Fraction of turns that are deliberately irrelevant noise"
    )
    max_turns_per_session_block: int = 8


class RolloutConfig(BaseModel):
    """Defaults for a rollout; individual fields overridable from the CLI."""

    num_steps: int = 100
    rng_seed: int = 42
    llm_model: str = "openai/gpt-4o-mini"
    llm_temperature: float = 0.7


class Constitution(BaseModel):
    """The full world definition: peers + relationships + rules + rollout defaults.

    A subset of the "world model" (per the design notes): governance rules and the
    seed relationship state, but not the transition dynamics themselves.
    """

    world_id: str
    peers: list[Peer]
    relationships: list[Relationship] = Field(default_factory=list)
    rules: InteractionRules = Field(default_factory=InteractionRules)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)

    def peer_ids(self) -> set[str]:
        return {p.id for p in self.peers}

    def get_peer(self, peer_id: str) -> Peer:
        for p in self.peers:
            if p.id == peer_id:
                return p
        raise KeyError(f"unknown peer: {peer_id}")

class Event(BaseModel):
    """One turn in the history. The trajectory is the full ordered list of these.
    Most are noise; a few carry planted gold facts."""

    step: int
    speaker: str
    content: str
    tags: list[str] = Field(default_factory=list)
    gold_fact_id: Optional[str] = Field(
        default=None, description="Set when this event plants a gold fact"
    )


class GoldFact(BaseModel):
    """One entry in the gold ledger — the answer key. Created *at plant time*, not
    extracted afterward, so it can never disagree with the trajectory."""

    fact_id: str
    step: int
    content: str = Field(..., description="Canonical statement of the fact, e.g. 'Alex adopted a cat named Miso'")
    answer: str = Field(..., description="The short expected answer for recall tasks, e.g. 'a cat named Miso'")
    subject: str = Field(..., description="Primary peer id the fact is about")
    kind: str = Field(..., description="Fact category, e.g. pet, relocation, job — used for negation task vocabulary")
    peer_refs: list[str] = Field(default_factory=list)
    checkpoint: bool = False

TaskType = Literal["fact_recall", "temporal", "relationship", "negation"] 

class Task(BaseModel):
    """A held-out question derived purely from the gold ledger + constitution."""

    task_id: str
    type: TaskType
    query: str
    expected: str
    required_fact_ids: list[str] = Field(default_factory=list)
    checkpoint_steps: list[int] = Field(default_factory=list)

class ToolCall(BaseModel):
    """A logged eval-agent tool invocation, used for trajectory verification."""

    task_id: str
    tool: str
    args: dict[str, Any]
    returned_event_steps: list[int] = Field(default_factory=list)


class TaskResult(BaseModel):
    """The agent's answer to one task plus the tool calls it made."""

    task_id: str
    type: TaskType
    query: str
    expected: str
    answer: str
    tool_calls: list[ToolCall] = Field(default_factory=list)


class Verdict(BaseModel):
    """Two-layer verification outcome for one task."""

    task_id: str
    type: TaskType
    end_state_pass: bool
    trajectory_hit: bool = Field(
        ..., description="Did the agent's tool calls surface events containing the required facts?"
    )
    checkpoint_recall: float = Field(
        ..., description="Fraction of required checkpoint steps the agent's queries touched"
    )
