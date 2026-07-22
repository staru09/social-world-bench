import json

from swb.rollout.engine import run_rollout
from swb.world.constitution import load_constitution, resolve_world


def _run(tmp_path, seed, name):
    cpath = resolve_world("generic_social_v1")
    constitution = load_constitution(cpath)
    out = tmp_path / name
    run_rollout(constitution, cpath, out_dir=out, num_steps=50, seed=seed, mode="scripted")
    return (out / "trajectory.jsonl").read_text(), json.loads((out / "gold_ledger.json").read_text())


def test_same_seed_identical_trajectory(tmp_path):
    traj_a, ledger_a = _run(tmp_path, 42, "a")
    traj_b, ledger_b = _run(tmp_path, 42, "b")
    assert traj_a == traj_b
    assert ledger_a == ledger_b


def test_different_seed_differs(tmp_path):
    traj_a, _ = _run(tmp_path, 42, "a")
    traj_c, _ = _run(tmp_path, 99, "c")
    # Facts land at fixed steps, but noise turns differ, so trajectories diverge.
    assert traj_a != traj_c


def test_gold_ledger_planted_and_tagged(tmp_path):
    _, ledger = _run(tmp_path, 42, "a")
    ids = {f["fact_id"] for f in ledger}
    # steps 5,20,35 are < 50 (planted); 60,80 are beyond and skipped.
    assert ids == {"f_pet", "f_job", "f_move"}
    checkpoints = {f["fact_id"] for f in ledger if f["checkpoint"]}
    assert checkpoints == {"f_pet", "f_job", "f_move"}
    for f in ledger:
        assert f["answer"].lower() in f["content"].lower()


def test_gold_events_carry_fact_id(tmp_path):
    cpath = resolve_world("generic_social_v1")
    constitution = load_constitution(cpath)
    out = tmp_path / "run"
    run_rollout(constitution, cpath, out_dir=out, num_steps=50, seed=42, mode="scripted")
    events = [json.loads(line) for line in (out / "trajectory.jsonl").read_text().splitlines()]
    gold_events = [e for e in events if e["gold_fact_id"]]
    assert len(gold_events) == 3
    # The event planting f_pet must actually contain the answer text.
    pet_event = next(e for e in gold_events if e["gold_fact_id"] == "f_pet")
    assert "miso" in pet_event["content"].lower()
