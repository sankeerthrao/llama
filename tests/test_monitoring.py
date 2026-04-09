# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""Tests for monitoring, metrics, and health checking."""

import json
import time

import pytest

from llama.monitoring import InferenceMetrics, StructuredLogger, HealthChecker


class TestInferenceMetrics:
    @pytest.fixture
    def metrics(self):
        return InferenceMetrics()

    def test_initial_state(self, metrics):
        summary = metrics.get_summary()
        assert summary["total_requests"] == 0
        assert summary["active_requests"] == 0
        assert summary["tokens_generated"] == 0

    def test_record_request(self, metrics):
        metrics.record_request(latency=0.5, tokens=100, success=True)
        summary = metrics.get_summary()
        assert summary["total_requests"] == 1
        assert summary["tokens_generated"] == 100

    def test_multiple_requests(self, metrics):
        for i in range(10):
            metrics.record_request(latency=0.1 * (i + 1), tokens=50, success=True)
        summary = metrics.get_summary()
        assert summary["total_requests"] == 10
        assert summary["tokens_generated"] == 500

    def test_record_error(self, metrics):
        metrics.record_error("timeout")
        metrics.record_error("timeout")
        metrics.record_error("oom")
        summary = metrics.get_summary()
        assert summary["errors"]["timeout"] == 2
        assert summary["errors"]["oom"] == 1

    def test_latency_percentiles(self, metrics):
        # Add known latencies
        for lat in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            metrics.record_request(latency=lat, tokens=10, success=True)
        summary = metrics.get_summary()
        # The implementation uses latency_p50/p95/p99 keys
        assert "latency_p50" in summary
        assert "latency_p95" in summary
        assert "latency_p99" in summary
        # p50 should be around 0.5-0.6
        assert 0.4 <= summary["latency_p50"] <= 0.7


class TestStructuredLogger:
    def test_create_logger(self):
        logger = StructuredLogger(name="test")
        assert logger is not None

    def test_create_logger_different_names(self):
        logger1 = StructuredLogger(name="test_a")
        logger2 = StructuredLogger(name="test_b")
        assert logger1 is not None
        assert logger2 is not None

    def test_info_no_crash(self):
        logger = StructuredLogger(name="test_info")
        # Should not raise
        logger.info("Test message", key1="value1", key2=42)

    def test_warning_no_crash(self):
        logger = StructuredLogger(name="test_warn")
        logger.warning("Warning message", detail="something")

    def test_error_no_crash(self):
        logger = StructuredLogger(name="test_err")
        logger.error("Error message", error_code=500)

    def test_debug_no_crash(self):
        logger = StructuredLogger(name="test_debug")
        logger.debug("Debug message", data={"x": 1})


class TestHealthChecker:
    @pytest.fixture
    def checker(self):
        return HealthChecker()

    def test_initial_state(self, checker):
        health = checker.check()
        assert health["status"] == "unhealthy"  # No model loaded
        assert health["model_loaded"] is False
        assert "uptime" in health or "uptime_seconds" in health

    def test_mark_model_loaded(self, checker):
        checker.mark_model_loaded()
        health = checker.check()
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True

    def test_record_inference(self, checker):
        checker.mark_model_loaded()
        checker.record_inference()
        health = checker.check()
        assert health["status"] == "healthy"

    def test_degraded_on_errors(self, checker):
        checker.mark_model_loaded()
        for _ in range(15):
            checker.record_error()
        health = checker.check()
        assert health["status"] == "degraded"

    def test_uptime_increases(self, checker):
        time.sleep(0.1)
        health = checker.check()
        uptime_key = "uptime" if "uptime" in health else "uptime_seconds"
        assert health[uptime_key] >= 0.05

    def test_device_info(self, checker):
        health = checker.check()
        assert "device_info" in health or "device" in health
