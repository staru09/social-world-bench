from swb.rollout.peer_sim import _clean
from swb.schema import Constitution, Event, GoldFact, Peer
from swb.verify.ledger_audit import audit_ledger

FACT = GoldFact(
    fact_id="f_move", step=3, content="relocating to Lisbon", answer="Lisbon",
    subject="jordan", kind="relocation", checkpoint=True,
)


def _events(*rows):
    return [Event(step=s, speaker=sp, content=c, gold_fact_id=g) for s, sp, c, g in rows]


def test_clean_run_passes_audit():
    events = _events(
        (0, "alex", "Nice weather today.", None),
        (3, "jordan", "I'm relocating to Lisbon in the fall.", "f_move"),
        (4, "sam", "My laptop fan is dying.", None),
    )
    assert audit_ledger(events, [FACT]).ok


def test_detects_leaked_answer_in_noise():
    events = _events(
        (1, "alex", "I hear Lisbon is lovely this time of year.", None),  # leak!
        (3, "jordan", "I'm relocating to Lisbon in the fall.", "f_move"),
    )
    report = audit_ledger(events, [FACT])
    assert not report.ok
    assert report.leaked_answers[0][0] == 1


def test_detects_negation_leak():
    events = _events(
        (0, "sam", "My neighbour got a new dog and it barks all night.", None),  # breaks neg task
        (3, "jordan", "I'm relocating to Lisbon in the fall.", "f_move"),
    )
    report = audit_ledger(events, [FACT])
    assert not report.ok
    assert report.negation_leaks[0][1] == "dog"


def test_negation_vocab_is_word_bounded():
    events = _events(
        (0, "sam", "I was dogged by a headache all day.", None),  # must NOT fire
        (3, "jordan", "I'm relocating to Lisbon in the fall.", "f_move"),
    )
    assert audit_ledger(events, [FACT]).ok


def test_detects_missing_plant():
    events = _events((0, "alex", "Just small talk.", None))
    report = audit_ledger(events, [FACT])
    assert not report.ok
    assert report.missing_plants == ["f_move"]


def test_clean_strips_echoed_speaker_prefix():
    c = Constitution(world_id="w", peers=[Peer(id="alex", name="Alex", persona="p")])
    assert _clean("alex: alex: hello there", "alex", c) == "hello there"
    assert _clean("Alex: hello there", "alex", c) == "hello there"
    assert _clean("hello there", "alex", c) == "hello there"
    # A colon that isn't a speaker label must survive.
    assert _clean("my plan: get coffee", "alex", c) == "my plan: get coffee"
