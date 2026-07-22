from swb.rollout.engine import run_rollout
from swb.tasks.generator import generate_tasks
from swb.world.constitution import load_constitution, resolve_world


def _rollout(tmp_path):
    cpath = resolve_world("generic_social_v1")
    constitution = load_constitution(cpath)
    out = tmp_path / "run"
    run_rollout(constitution, cpath, out_dir=out, num_steps=100, seed=42, mode="scripted")
    return out, constitution


def test_generates_all_task_types(tmp_path):
    out, constitution = _rollout(tmp_path)
    tasks = generate_tasks(out / "gold_ledger.json", constitution)
    types = {t.type for t in tasks}
    assert {"fact_recall", "temporal", "relationship", "negation"} <= types


def test_fact_recall_expected_matches_ledger(tmp_path):
    out, constitution = _rollout(tmp_path)
    tasks = generate_tasks(out / "gold_ledger.json", constitution)
    pet = next(t for t in tasks if t.task_id == "recall_f_pet")
    assert pet.expected == "a cat named Miso"
    assert pet.required_fact_ids == ["f_pet"]
    assert "Alex" in pet.query


def test_temporal_orders_earliest_first(tmp_path):
    out, constitution = _rollout(tmp_path)
    tasks = generate_tasks(out / "gold_ledger.json", constitution)
    temporal = next(t for t in tasks if t.type == "temporal")
    # f_pet at step 5 (Alex) is earliest, so the expected first-sharer is Alex.
    assert temporal.expected == "Alex"


def test_negation_expected_no(tmp_path):
    out, constitution = _rollout(tmp_path)
    tasks = generate_tasks(out / "gold_ledger.json", constitution)
    negs = [t for t in tasks if t.type == "negation"]
    assert negs
    assert all(t.expected == "no" for t in negs)


def test_relationship_task(tmp_path):
    out, constitution = _rollout(tmp_path)
    tasks = generate_tasks(out / "gold_ledger.json", constitution)
    rel = next(t for t in tasks if t.type == "relationship")
    assert rel.expected in {"Alex", "Sam", "Jordan"}
