"""Fixed-work carbon-deferral head-to-head — the fair version of the ledger.

The serverless-training-ledger harness compared a carbon-aware run that *dropped*
the dirty-window work against an always-on run, so the two did unequal work. This
harness fixes that: both policies complete the **same total iterations**, so the
comparison is honest. The cost of being green is then paid in *makespan*, not in
unfinished work.

Both run a real resnet18/CIFAR-10 job on the host GPU, metered marginally (the
co-tenant background is subtracted), with carbon billed at each tick's intensity:

  - always-on   : never pauses; finishes in the fewest ticks.
  - carbon-aware : pauses (checkpoint + scale-to-zero) while intensity is above a
                   threshold and defers that work to later clean ticks; finishes
                   later but does more of its work under low intensity, paying a
                   real resume cost (reload + CUDA reinit + warmup) per wake-up.

The schedule repeats until the work is done, so carbon-aware genuinely catches
up. Output is the carbon-vs-makespan trade-off, per freed-GPU regime
(dedicated / reallocated / powered-down), plus the equal-work break-even window.

Requires a real GPU; a Knative service is optional (``--no-drive-pod`` runs the
host-side study alone, since WS-level scale-to-zero fidelity is shown by the
serverless-training-ledger harness).

Usage::

    python -m experiments.exp_carbon_deferral_breakeven \
        --total-iters 4000 --tick-seconds 3 --intensities 200,200,900,900 \
        --threshold 800 --out artifacts/deferral_breakeven.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from rich.console import Console

from hasagi.energy.marginal_meter import MarginalEnergyMeter
from hasagi.energy.pod_ledger import (
    PHASE_ACTIVE,
    PHASE_COLD_START,
    PHASE_IDLE,
    LedgerReport,
    PodEnergyLedger,
)
from hasagi.energy.regime import GpuRegime, break_even_window_s, regime_breakdown
from hasagi.pool.knative_pool import KnativePool
from hasagi.worker.host_trainer import HostTrainer


def _train_until(trainer: HostTrainer, seconds: float, total_iters: int) -> None:
    """Train in small chunks until ``seconds`` elapse or ``total_iters`` reached."""
    start = time.monotonic()
    while time.monotonic() - start < seconds and trainer.iters_done < total_iters:
        trainer.train_iters_count(8)


def _run_policy(
    console: Console,
    name: str,
    *,
    ckpt_path: str,
    energy_fn,
    schedule: list[float],
    threshold: float,
    total_iters: int,
    tick_seconds: float,
    pool: KnativePool | None,
    max_ticks: int,
) -> dict:
    """Run one policy to completion (``total_iters``); return its ledger + stats.

    ``threshold = inf`` makes the policy never pause (the always-on baseline).
    """
    console.print(f"[bold]Policy: {name}[/] (target {total_iters} iters)")
    trainer = HostTrainer(ckpt_path=ckpt_path)
    ledger = PodEnergyLedger(energy_fn)
    running = False
    ever_started = False
    pauses = 0
    ticks = 0
    t0 = time.monotonic()

    while trainer.iters_done < total_iters and ticks < max_ticks:
        intensity = schedule[ticks % len(schedule)]
        ticks += 1
        pause = intensity > threshold
        if not pause:
            if not running:
                ledger.mark(PHASE_COLD_START, intensity)
                if pool is not None:
                    pool.scale(target=1, timeout_seconds=30.0, wait_for_ready=False)
                if not ever_started:
                    trainer.cold_init()
                    ever_started = True
                else:
                    trainer.resume()
                running = True
            ledger.mark(PHASE_ACTIVE, intensity)
            _train_until(trainer, tick_seconds, total_iters)
        else:
            if running:
                trainer.checkpoint()
                trainer.teardown()
                if pool is not None:
                    pool.scale(target=0, timeout_seconds=10.0, wait_for_ready=False)
                running = False
                pauses += 1
            ledger.mark(PHASE_IDLE, intensity)
            time.sleep(tick_seconds)

    makespan_s = time.monotonic() - t0
    report = ledger.report()
    if pool is not None:
        pool.scale(target=0, timeout_seconds=10.0, wait_for_ready=False)
    console.print(
        f"  done: {trainer.iters_done} iters in {ticks} ticks, {pauses} pause(s), "
        f"makespan {makespan_s:.1f}s; marginal active power "
        f"≈ {_phase_power_w(report, PHASE_ACTIVE):.0f} W"
    )
    return {
        "name": name,
        "report": report,
        "iters": trainer.iters_done,
        "ticks": ticks,
        "pauses": pauses,
        "makespan_s": makespan_s,
    }


def _phase_power_w(rep: LedgerReport, phase: str) -> float:
    e_kwh = rep.energy_by_phase_kwh.get(phase, 0.0)
    dur_s = rep.duration_by_phase_s.get(phase, 0.0)
    return e_kwh * 3_600_000.0 / dur_s if dur_s > 0 else 0.0


def run(args: argparse.Namespace) -> int:
    console = Console()
    schedule = [float(x) for x in args.intensities.split(",")]
    dirty, clean = max(schedule), min(schedule)
    console.print(
        f"[bold]Fixed-work carbon-deferral break-even[/] — {args.total_iters} iters, "
        f"tick {args.tick_seconds}s, pause above {args.threshold:.0f} gCO2/kWh; "
        f"schedule={schedule}"
    )

    meter = MarginalEnergyMeter(device_index=args.device, poll_interval_ms=100)
    bg_mean, bg_sd = meter.calibrate(seconds=args.calibrate_s)
    console.print(f"[bold]Background[/]: {bg_mean:.1f} W (sd {bg_sd:.1f} W) subtracted.")
    meter.start()
    energy_fn = meter.cumulative_kwh

    pool = None if args.no_drive_pod else KnativePool(service=args.service, namespace=args.namespace)
    try:
        base = _run_policy(
            console, "always-on", ckpt_path="./artifacts/deferral_base_ckpt.pt",
            energy_fn=energy_fn, schedule=schedule, threshold=math.inf,
            total_iters=args.total_iters, tick_seconds=args.tick_seconds,
            pool=pool, max_ticks=args.max_ticks,
        )
        aware = _run_policy(
            console, "carbon-aware", ckpt_path="./artifacts/deferral_aware_ckpt.pt",
            energy_fn=energy_fn, schedule=schedule, threshold=args.threshold,
            total_iters=args.total_iters, tick_seconds=args.tick_seconds,
            pool=pool, max_ticks=args.max_ticks,
        )
    finally:
        meter.stop()

    active_w = _phase_power_w(base["report"], PHASE_ACTIVE)
    resume_kwh = aware["report"].resume_energy_kwh

    console.print("\n[bold]Equal-work carbon vs makespan, by freed-GPU regime[/]:")
    base_bd = regime_breakdown(base["report"], dedicated_idle_w=args.dedicated_idle_w)
    aware_bd = regime_breakdown(aware["report"], dedicated_idle_w=args.dedicated_idle_w)
    regimes_out: dict[str, dict] = {}
    for regime in GpuRegime:
        b = base_bd[regime.value].total_carbon_g
        a = aware_bd[regime.value].total_carbon_g
        delta = a - b
        rel = (100.0 * delta / b) if b else 0.0
        idle_w = args.dedicated_idle_w if regime is GpuRegime.DEDICATED else 0.0
        t_star = break_even_window_s(
            active_power_w=active_w, intensity_dirty=dirty, intensity_clean=clean,
            resume_energy_kwh=resume_kwh, idle_power_w=idle_w,
        )
        verdict = "SAVES" if delta < 0 else "LOSES"
        t_str = "never" if t_star == math.inf else f"{t_star:.1f}s"
        console.print(
            f"  [{regime.value:12s}] carbon-aware {a:.3f} vs always-on {b:.3f} gCO2 "
            f"→ {delta:+.3f} ({rel:+.1f}%) {verdict}; break-even tick {t_str}"
        )
        regimes_out[regime.value] = {
            "carbon_aware_g": a, "always_on_g": b,
            "delta_g": delta, "delta_pct": rel,
            "break_even_window_s": None if t_star == math.inf else t_star,
            "idle_power_w_assumed": idle_w,
        }
    makespan_cost = aware["makespan_s"] - base["makespan_s"]
    console.print(
        f"[bold]Makespan cost of going green[/]: "
        f"+{makespan_cost:.1f}s ({aware['makespan_s']:.1f}s vs {base['makespan_s']:.1f}s); "
        f"{aware['pauses']} pause(s), {aware['ticks']} vs {base['ticks']} ticks."
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schedule_g_per_kwh": schedule,
            "threshold_g_per_kwh": args.threshold,
            "total_iters": args.total_iters,
            "tick_seconds": args.tick_seconds,
            "trace_source": "synthetic-parametric",
            "energy_source": "nvml-measured-marginal",
            "background_w_mean": bg_mean,
            "background_w_sd": bg_sd,
            "active_marginal_power_w": active_w,
            "resume_energy_wh": resume_kwh * 1000.0,
            "dedicated_idle_w_assumed": args.dedicated_idle_w,
            "always_on": {k: v for k, v in base.items() if k != "report"},
            "carbon_aware": {k: v for k, v in aware.items() if k != "report"},
            "makespan_cost_s": makespan_cost,
            "regimes": regimes_out,
        }, indent=2))
        console.print(f"[dim]wrote {out}[/]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--service", default="hasagi-worker-lifecycle")
    p.add_argument("--namespace", default="hasagi-validation")
    p.add_argument("--no-drive-pod", action="store_true",
                   help="Skip the Knative pod; run the host-side study alone.")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--intensities", default="200,200,900,900",
                   help="Comma-separated per-tick grid intensity gCO2/kWh; cycled until done.")
    p.add_argument("--threshold", type=float, default=800.0)
    p.add_argument("--total-iters", type=int, default=4000)
    p.add_argument("--tick-seconds", type=float, default=3.0)
    p.add_argument("--max-ticks", type=int, default=400,
                   help="Safety cap so an over-aggressive threshold cannot loop forever.")
    p.add_argument("--calibrate-s", type=float, default=6.0)
    p.add_argument("--dedicated-idle-w", type=float, default=30.0,
                   help="Idle floor charged in the dedicated regime; measure on a clean GPU.")
    p.add_argument("--out", default=None)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
