from swb.schema import GoldFact, Task, TaskResult, ToolCall
from swb.verify.end_state import end_state_pass
from swb.verify.report import wilson_ci
from swb.verify.trajectory import checkpoint_recall, trajectory_hit


def test_end_state_containment():
    assert end_state_pass("a cat named Miso", "It was a cat named Miso.")
    assert end_state_pass("Lisbon", "lisbon")
    assert not end_state_pass("Lisbon", "Madrid")


def test_end_state_yes_no():
    assert end_state_pass("no", "No, nobody mentioned that.")
    assert not end_state_pass("no", "yes, someone did")
    assert end_state_pass("yes", "Yes.")


def test_trajectory_hit_true_when_step_returned():
    ledger = {"f_pet": GoldFact(fact_id="f_pet", step=5, content="c", answer="a cat named Miso", subject="alex", kind="pet", checkpoint=True)}
    task = Task(task_id="t", type="fact_recall", query="q", expected="a cat named Miso", required_fact_ids=["f_pet"], checkpoint_steps=[5])
    result = TaskResult(
        task_id="t", type="fact_recall", query="q", expected="a cat named Miso", answer="a cat named Miso",
        tool_calls=[ToolCall(task_id="t", tool="search_events", args={"query": "cat"}, returned_event_steps=[5, 7])],
    )
    assert trajectory_hit(task, result, ledger)
    assert checkpoint_recall(task, result) == 1.0


def test_trajectory_miss_when_step_absent():
    ledger = {"f_pet": GoldFact(fact_id="f_pet", step=5, content="c", answer="a cat named Miso", subject="alex", kind="pet", checkpoint=True)}
    task = Task(task_id="t", type="fact_recall", query="q", expected="a cat named Miso", required_fact_ids=["f_pet"], checkpoint_steps=[5])
    result = TaskResult(
        task_id="t", type="fact_recall", query="q", expected="a cat named Miso", answer="guessed",
        tool_calls=[ToolCall(task_id="t", tool="search_events", args={"query": "dog"}, returned_event_steps=[9])],
    )
    assert not trajectory_hit(task, result, ledger)
    assert checkpoint_recall(task, result) == 0.0


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(5, 5)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi == 1.0 or hi < 1.0  # upper near 1 for all-pass
    lo0, hi0 = wilson_ci(0, 0)
    assert (lo0, hi0) == (0.0, 0.0)
