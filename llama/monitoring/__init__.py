"""Monitoring infrastructure for LLaMA inference serving.

Provides metrics collection, structured logging, and health checking
that work on both CPU-only and CUDA-enabled machines.

Quick start::

    from llama.monitoring import InferenceMetrics, StructuredLogger, HealthChecker
    from llama.monitoring.logging import setup_logging

    logger = setup_logging(level="INFO", format="json")
    metrics = InferenceMetrics()
    health = HealthChecker()
"""

from llama.monitoring.health import HealthChecker
from llama.monitoring.logging import StructuredLogger, setup_logging
from llama.monitoring.metrics import InferenceMetrics

__all__ = [
    "InferenceMetrics",
    "StructuredLogger",
    "HealthChecker",
    "setup_logging",
]
