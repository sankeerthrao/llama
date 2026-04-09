"""Inference metrics tracking for LLaMA model serving.

Provides thread-safe counters, gauges, and latency histograms for monitoring
inference workloads. Optionally integrates with prometheus_client if available.
"""

import statistics
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


class InferenceMetrics:
    """Thread-safe inference metrics collector.

    Tracks request counts, latencies, token throughput, error rates,
    and cache performance for LLaMA inference serving.

    Example::

        metrics = InferenceMetrics()
        metrics.record_request(latency=0.35, tokens=128, success=True)
        summary = metrics.get_summary()
    """

    def __init__(self, enable_prometheus: bool = False) -> None:
        """Initialise metrics stores.

        Args:
            enable_prometheus: If True and prometheus_client is installed,
                register Prometheus collectors alongside internal tracking.
        """
        self._lock = threading.Lock()

        # Internal counters / gauges
        self.total_requests: int = 0
        self.active_requests: int = 0
        self.tokens_generated: int = 0
        self.request_latency_seconds: List[float] = []
        self.errors: Dict[str, int] = defaultdict(int)

        # Cache tracking
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Prometheus collectors (optional)
        self._prom_enabled = enable_prometheus and _PROMETHEUS_AVAILABLE
        if self._prom_enabled:
            self._prom_total_requests = Counter(
                "llama_inference_requests_total",
                "Total inference requests",
            )
            self._prom_active_requests = Gauge(
                "llama_inference_active_requests",
                "Currently active inference requests",
            )
            self._prom_latency = Histogram(
                "llama_inference_latency_seconds",
                "Inference request latency in seconds",
                buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )
            self._prom_tokens = Counter(
                "llama_inference_tokens_generated_total",
                "Total tokens generated",
            )
            self._prom_errors = Counter(
                "llama_inference_errors_total",
                "Total inference errors",
                ["error_type"],
            )

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    def record_request(
        self,
        latency: float,
        tokens: int,
        success: bool = True,
    ) -> None:
        """Record a completed inference request.

        Args:
            latency: Wall-clock time in seconds for the request.
            tokens: Number of tokens generated.
            success: Whether the request completed without error.
        """
        with self._lock:
            self.total_requests += 1
            self.request_latency_seconds.append(latency)
            self.tokens_generated += tokens

            if not success:
                self.errors["request_failure"] += 1

        if self._prom_enabled:
            self._prom_total_requests.inc()
            self._prom_latency.observe(latency)
            self._prom_tokens.inc(tokens)
            if not success:
                self._prom_errors.labels(error_type="request_failure").inc()

    def record_error(self, error_type: str) -> None:
        """Record an error by type.

        Args:
            error_type: Categorical label for the error (e.g. ``"timeout"``,
                ``"oom"``, ``"invalid_input"``).
        """
        with self._lock:
            self.errors[error_type] += 1

        if self._prom_enabled:
            self._prom_errors.labels(error_type=error_type).inc()

    def record_cache_access(self, hit: bool) -> None:
        """Record a KV-cache lookup.

        Args:
            hit: ``True`` if the cache entry was found, ``False`` otherwise.
        """
        with self._lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def increment_active(self) -> None:
        """Increment the active-request gauge (call when a request starts)."""
        with self._lock:
            self.active_requests += 1
        if self._prom_enabled:
            self._prom_active_requests.inc()

    def decrement_active(self) -> None:
        """Decrement the active-request gauge (call when a request finishes)."""
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
        if self._prom_enabled:
            self._prom_active_requests.dec()

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def tokens_per_second(self) -> float:
        """Average tokens generated per second across all recorded requests."""
        with self._lock:
            total_latency = sum(self.request_latency_seconds)
            if total_latency == 0:
                return 0.0
            return self.tokens_generated / total_latency

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of cache accesses that were hits (0.0–1.0)."""
        with self._lock:
            total = self._cache_hits + self._cache_misses
            if total == 0:
                return 0.0
            return self._cache_hits / total

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _percentile(self, data: List[float], pct: float) -> float:
        """Compute a percentile from a sorted list of values."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (pct / 100.0)
        floor_k = int(k)
        ceil_k = min(floor_k + 1, len(sorted_data) - 1)
        weight = k - floor_k
        return sorted_data[floor_k] * (1 - weight) + sorted_data[ceil_k] * weight

    def get_summary(self) -> Dict:
        """Return a snapshot of all current metrics.

        Returns:
            dict with keys: total_requests, active_requests,
            tokens_generated, tokens_per_second, cache_hit_rate,
            latency_p50, latency_p95, latency_p99, latency_mean,
            errors.
        """
        with self._lock:
            latencies = list(self.request_latency_seconds)
            errors_snapshot = dict(self.errors)
            summary: Dict = {
                "total_requests": self.total_requests,
                "active_requests": self.active_requests,
                "tokens_generated": self.tokens_generated,
                "tokens_per_second": (
                    self.tokens_generated / sum(latencies) if latencies else 0.0
                ),
                "cache_hit_rate": (
                    self._cache_hits / (self._cache_hits + self._cache_misses)
                    if (self._cache_hits + self._cache_misses) > 0
                    else 0.0
                ),
                "latency_p50": self._percentile(latencies, 50),
                "latency_p95": self._percentile(latencies, 95),
                "latency_p99": self._percentile(latencies, 99),
                "latency_mean": (
                    statistics.mean(latencies) if latencies else 0.0
                ),
                "errors": errors_snapshot,
            }
        return summary
