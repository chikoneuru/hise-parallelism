"""Unit tests for per-stage 1F1B WRR scheduler."""
from __future__ import annotations

from hise.parallel.inter_batch import InterBatchScheduler, Node, weights_for_stage


def test_weights_sum_to_one() -> None:
    nodes = [
        Node(node_id="w1", stage_id=0, capacity_flops=1.0),
        Node(node_id="w2", stage_id=0, capacity_flops=3.0),
    ]
    w = weights_for_stage(nodes)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["w2"] > w["w1"]


def test_backward_takes_priority() -> None:
    sched = InterBatchScheduler([
        Node(node_id="w1", stage_id=1, capacity_flops=1.0),
        Node(node_id="w2", stage_id=1, capacity_flops=1.0),
    ], is_last_stage=False)
    sched.enqueue_forward(1)
    sched.enqueue_backward(2, target_node_id="w1")
    events = sched.tick()
    assert any(d == "bwd" for _, d, _ in events)


def test_last_stage_auto_triggers_backward() -> None:
    sched = InterBatchScheduler(
        [Node(node_id="w1", stage_id=3, capacity_flops=1.0)],
        is_last_stage=True,
    )
    sched.enqueue_forward(42)
    events1 = sched.tick()
    assert events1 == [("w1", "fwd", 42)]
    events2 = sched.tick()
    assert events2 == [("w1", "bwd", 42)]


def test_non_last_stage_no_auto_backward() -> None:
    sched = InterBatchScheduler(
        [Node(node_id="w1", stage_id=0, capacity_flops=1.0)],
        is_last_stage=False,
    )
    sched.enqueue_forward(42)
    sched.tick()
    assert sched.pending() == 0


def test_proportional_dispatch() -> None:
    sched = InterBatchScheduler([
        Node(node_id="w1", stage_id=0, capacity_flops=1.0),
        Node(node_id="w2", stage_id=0, capacity_flops=3.0),
    ], is_last_stage=False)
    for mb in range(40):
        sched.enqueue_forward(mb)
    n1 = len(next(n.fwd_queue for n in sched.nodes if n.node_id == "w1"))
    n2 = len(next(n.fwd_queue for n in sched.nodes if n.node_id == "w2"))
    assert abs(n2 - 3 * n1) < 6
