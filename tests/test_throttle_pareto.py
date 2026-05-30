"""Tests for the carbon-throttle vs pause vs always-on Pareto simulator."""
from __future__ import annotations

import pytest

from hasagi.energy.throttle_pareto import (
    CapPoint,
    PowerCapProfile,
    pareto,
    simulate_policy,
)

# Synthetic profile: energy-optimal at 200 W (4.5 J/it), throughput-max at 300 W.
_PROFILE = PowerCapProfile(
    gpu_name="synthetic",
    points={
        100.0: CapPoint(100.0, 10.0, 100.0, 10.0 / 3_600_000.0),
        200.0: CapPoint(200.0, 40.0, 180.0, 4.5 / 3_600_000.0),
        300.0: CapPoint(300.0, 50.0, 270.0, 5.4 / 3_600_000.0),
    },
)

_SCHED = [200.0, 800.0]   # clean, dirty
_KW = dict(total_iters=1000, window_s=10.0, schedule_g=_SCHED, threshold_g=500.0)


def test_profile_optima() -> None:
    assert _PROFILE.energy_optimal_cap == 200.0
    assert _PROFILE.max_throughput_cap == 300.0
    assert _PROFILE.point(195.0).cap_w == 200.0   # nearest


def test_always_on_pays_dirty_intensity() -> None:
    r = simulate_policy(_PROFILE, name="always-on", clean_cap_w=300.0, dirty_cap_w=300.0, **_KW)
    # 500 iters clean@200 + 500 iters dirty@800, each 7.5e-4 kWh
    assert r.total_carbon_g == pytest.approx(7.5e-4 * 200 + 7.5e-4 * 800)   # 0.75 g
    assert r.makespan_s == pytest.approx(20.0)
    assert r.iters == 1000


def test_pause_defers_dirty_work_zero_idle() -> None:
    r = simulate_policy(
        _PROFILE, name="pause", clean_cap_w=300.0, dirty_cap_w=None,
        idle_power_w=0.0, **_KW,
    )
    # all work done in clean windows at 200 → 2×0.15 g; +10 s makespan for the pause
    assert r.total_carbon_g == pytest.approx(0.30)
    assert r.makespan_s == pytest.approx(30.0)
    assert r.idle_carbon_g == 0.0


def test_pause_dedicated_idle_adds_carbon() -> None:
    r = simulate_policy(
        _PROFILE, name="pause", clean_cap_w=300.0, dirty_cap_w=None,
        idle_power_w=50.0, **_KW,
    )
    idle_g = 50.0 * 10.0 / 3_600_000.0 * 800.0   # one dirty window idle @ 800
    assert r.idle_carbon_g == pytest.approx(idle_g)
    assert r.total_carbon_g == pytest.approx(0.30 + idle_g)


def test_pause_charges_resume_on_wakeup() -> None:
    r = simulate_policy(
        _PROFILE, name="pause", clean_cap_w=300.0, dirty_cap_w=None,
        resume_energy_kwh=1e-4, **_KW,
    )
    # one wake-up (the clean window after the dirty pause), billed at clean 200
    assert r.resume_carbon_g == pytest.approx(1e-4 * 200.0)


def test_throttle_between_alwayson_and_pause() -> None:
    r = simulate_policy(
        _PROFILE, name="throttle", clean_cap_w=300.0, dirty_cap_w=200.0, **_KW,
    )
    # clean 500@200 (0.15) + dirty 400@200cap@800 (0.4) + clean 100@200 (0.03)
    assert r.total_carbon_g == pytest.approx(0.58)
    assert r.makespan_s == pytest.approx(22.0)


def test_pareto_orders_carbon_and_makespan() -> None:
    res = pareto(_PROFILE, total_iters=1000, window_s=10.0, schedule_g=_SCHED, threshold_g=500.0)
    assert set(res) == {"always_on", "pause", "throttle"}
    # pause: least carbon, most makespan; always-on: most carbon, least makespan
    assert res["pause"].total_carbon_g < res["throttle"].total_carbon_g < res["always_on"].total_carbon_g
    assert res["pause"].makespan_s > res["always_on"].makespan_s
    # all complete the same work
    for r in res.values():
        assert r.iters >= 1000
