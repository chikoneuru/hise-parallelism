"""Unit tests for ElasticFlow MSS + HISE Energy-Budgeted MSS."""
from __future__ import annotations

from hise.admission.mss import (
    EnergyBudgetMSS,
    ScalingCurve,
    greedy_marginal_allocation,
    minimum_satisfactory_share,
)


def _concave_curve(max_gpus: int = 16) -> ScalingCurve:
    # 1, 1.82, 2.55, ..., concave shape (x^0.85).
    return ScalingCurve(throughput_per_gpu_count=[x ** 0.85 for x in range(1, max_gpus + 1)])


def test_mss_returns_minimal_gpu_count() -> None:
    curve = _concave_curve()
    # 100 iters in 100 seconds: need throughput >= 1 → 1 gpu is enough.
    assert minimum_satisfactory_share(100, 100.0, curve) == 1
    # 1000 iters in 100 seconds: need throughput >= 10 → ~14 gpus (since 14^0.85 ≈ 9.4, 16^0.85 ≈ 10.5).
    mss = minimum_satisfactory_share(1000, 100.0, curve)
    assert mss > 1
    assert curve.throughput(mss) >= 10.0
    assert curve.throughput(mss - 1) < 10.0


def test_mss_returns_zero_when_infeasible() -> None:
    curve = _concave_curve(max_gpus=4)
    # 10 000 iters in 1 sec: nothing fits in 4-gpu cluster.
    assert minimum_satisfactory_share(10_000, 1.0, curve) == 0


def test_energy_budget_mss_admits_with_generous_budget() -> None:
    curve = _concave_curve()
    # 500 iters at concave curve completes well under 200s with 1 GPU; energy is small.
    ebmss = EnergyBudgetMSS(
        curve=curve,
        power_per_gpu_w=300.0,
        energy_budget_kwh=1.0,  # 1 kWh — plenty for a 500-iter mini-job.
    )
    decision = ebmss.find(iterations_remaining=500, deadline_seconds=200.0)
    assert decision.admitted
    assert decision.gpus >= 1


def test_energy_budget_mss_rejects_when_energy_too_low() -> None:
    curve = _concave_curve()
    # Tiny energy budget but generous deadline → no allocation fits the energy box.
    ebmss = EnergyBudgetMSS(
        curve=curve,
        power_per_gpu_w=300.0,
        energy_budget_kwh=1e-9,
    )
    decision = ebmss.find(iterations_remaining=10_000, deadline_seconds=3600.0)
    assert not decision.admitted


def test_energy_budget_mss_with_carbon_proxy_secondary() -> None:
    curve = _concave_curve()
    # Energy budget admits us; carbon proxy adds a (loose) secondary constraint.
    ebmss = EnergyBudgetMSS(
        curve=curve,
        power_per_gpu_w=300.0,
        energy_budget_kwh=1.0,
        carbon_intensity_forecast=lambda t: 100.0,  # clean grid
        carbon_budget_g=1_000.0,
    )
    decision = ebmss.find(iterations_remaining=500, deadline_seconds=200.0)
    assert decision.admitted
    assert "carbon proxy" in decision.reason


def test_energy_budget_mss_rejects_when_carbon_too_tight() -> None:
    curve = _concave_curve()
    # Deadline + energy are both fine, but carbon proxy is impossibly tight — the carbon
    # branch should be what causes rejection (not deadline / not energy).
    ebmss = EnergyBudgetMSS(
        curve=curve,
        power_per_gpu_w=300.0,
        energy_budget_kwh=100.0,
        carbon_intensity_forecast=lambda t: 5_000.0,
        carbon_budget_g=0.001,
    )
    decision = ebmss.find(iterations_remaining=500, deadline_seconds=3600.0)
    assert not decision.admitted
    assert "energy budget" in decision.reason


def test_greedy_marginal_allocation_distributes_remaining() -> None:
    curve = _concave_curve()
    admitted = [("job-a", curve, 2), ("job-b", curve, 3)]
    alloc = greedy_marginal_allocation(admitted, available_gpus=10)
    assert alloc["job-a"] >= 2
    assert alloc["job-b"] >= 3
    assert alloc["job-a"] + alloc["job-b"] == 10
