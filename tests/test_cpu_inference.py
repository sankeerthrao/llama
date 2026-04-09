# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""Tests for CPU inference engine."""

import pytest
import torch

from llama.cpu_inference import (
    CPUInferenceConfig,
    CPUModelLoader,
    get_device_info,
    get_optimal_device,
)


class TestGetOptimalDevice:
    def test_returns_string(self):
        device = get_optimal_device()
        assert isinstance(device, str)
        assert device in ("cuda", "mps", "cpu")

    def test_cpu_when_no_gpu(self):
        # In CI/test environment without GPU, should return cpu
        if not torch.cuda.is_available():
            assert get_optimal_device() == "cpu"


class TestGetDeviceInfo:
    def test_returns_dict(self):
        info = get_device_info()
        assert isinstance(info, dict)
        assert "selected_device" in info
        assert "cuda_available" in info
        assert "cpu_threads" in info
        assert "torch_version" in info

    def test_cpu_threads_positive(self):
        info = get_device_info()
        assert info["cpu_threads"] > 0

    def test_torch_version(self):
        info = get_device_info()
        assert info["torch_version"] == torch.__version__


class TestCPUInferenceConfig:
    def test_defaults(self):
        cfg = CPUInferenceConfig()
        assert cfg.num_threads == 0
        assert cfg.use_torch_compile is False
        assert cfg.dtype == "float32"
        assert cfg.enable_memory_efficient is True

    def test_custom(self):
        cfg = CPUInferenceConfig(num_threads=4, dtype="bfloat16")
        assert cfg.num_threads == 4
        assert cfg.dtype == "bfloat16"


class TestCPUModelLoader:
    def test_init_default(self):
        loader = CPUModelLoader()
        assert loader.config.num_threads == 0

    def test_init_custom(self):
        cfg = CPUInferenceConfig(num_threads=2)
        loader = CPUModelLoader(cfg)
        assert loader.config.num_threads == 2

    def test_get_dtype(self):
        loader = CPUModelLoader(CPUInferenceConfig(dtype="float32"))
        assert loader._get_dtype() == torch.float32

        loader = CPUModelLoader(CPUInferenceConfig(dtype="bfloat16"))
        assert loader._get_dtype() == torch.bfloat16

    def test_thread_setup(self):
        cfg = CPUInferenceConfig(num_threads=4)
        CPUModelLoader(cfg)
        assert torch.get_num_threads() == 4

    def test_auto_threads(self):
        import os
        cfg = CPUInferenceConfig(num_threads=0)
        CPUModelLoader(cfg)
        # Should set to cpu_count
        expected = os.cpu_count() or 4
        assert torch.get_num_threads() == expected
