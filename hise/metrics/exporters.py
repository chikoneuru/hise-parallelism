"""Shared Prometheus collectors. Imported by orchestrator + worker so labels stay consistent."""
from __future__ import annotations

from prometheus_client import Counter, Gauge

JOB_ALLOCATED_GPUS = Gauge("hise_job_allocated_gpus", "GPUs allocated per job", ["job_id"])
THROUGHPUT = Gauge("hise_job_throughput_iter_per_s", "Iter/s per job", ["job_id"])
LOSS = Gauge("hise_job_loss", "Latest training loss", ["job_id"])
GPU_UTIL = Gauge("hise_worker_gpu_util", "Approx GPU utilisation (0..1)", ["worker_id"])
ITERATIONS_TOTAL = Counter("hise_job_iterations_total", "Iterations processed", ["job_id"])
