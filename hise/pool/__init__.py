"""GPU Burst Pool Manager — abstracts worker lifecycle across backends."""
from hise.pool.worker_registry import WorkerInfo, WorkerRegistry

__all__ = ["WorkerInfo", "WorkerRegistry"]
