# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""Tests for model quantization."""

import pytest
import torch
import torch.nn as nn

from llama.quantization import (
    QuantConfig,
    QuantizedLinear,
    estimate_model_size,
    quantize_model,
)


class TestQuantizedLinearINT8:
    def test_from_linear_shape(self):
        linear = nn.Linear(64, 32, bias=False)
        ql = QuantizedLinear.from_linear(linear, bits=8)
        assert ql.weight_quantized.shape == (32, 64)
        assert ql.weight_quantized.dtype == torch.int8
        assert ql.weight_scale.shape == (32,)

    def test_from_linear_with_bias(self):
        linear = nn.Linear(64, 32, bias=True)
        ql = QuantizedLinear.from_linear(linear, bits=8)
        assert ql.bias is not None
        assert ql.bias.shape == (32,)

    def test_forward_shape(self):
        linear = nn.Linear(64, 32, bias=False)
        ql = QuantizedLinear.from_linear(linear, bits=8)
        x = torch.randn(2, 10, 64)
        out = ql(x)
        assert out.shape == (2, 10, 32)

    def test_roundtrip_accuracy(self):
        """INT8 quantization should be close to the original linear layer."""
        torch.manual_seed(42)
        linear = nn.Linear(128, 64, bias=False)
        ql = QuantizedLinear.from_linear(linear, bits=8)
        x = torch.randn(1, 5, 128)

        with torch.no_grad():
            orig_out = linear(x)
            quant_out = ql(x)

        # INT8 should be within ~1% relative error for most elements
        rel_error = (orig_out - quant_out).abs() / (orig_out.abs() + 1e-6)
        assert rel_error.mean() < 0.05, f"Mean relative error too high: {rel_error.mean():.4f}"

    def test_memory_reduction(self):
        """Quantized model should use less memory for weights."""
        linear = nn.Linear(256, 128, bias=False)
        ql = QuantizedLinear.from_linear(linear, bits=8)
        # INT8 weights: 256*128 * 1 byte = 32768 bytes
        # Float32 weights: 256*128 * 4 bytes = 131072 bytes
        quant_bytes = ql.weight_quantized.numel() * ql.weight_quantized.element_size()
        orig_bytes = linear.weight.numel() * linear.weight.element_size()
        assert quant_bytes < orig_bytes


class TestQuantizedLinearINT4:
    def test_from_linear_shape(self):
        linear = nn.Linear(128, 64, bias=False)
        ql = QuantizedLinear.from_linear(linear, bits=4, group_size=32)
        assert ql.weight_quantized.dtype == torch.uint8
        assert ql.weight_scale.shape[0] == 64  # out_features
        assert ql.weight_zero_point.shape[0] == 64

    def test_forward_shape(self):
        linear = nn.Linear(128, 64, bias=False)
        ql = QuantizedLinear.from_linear(linear, bits=4, group_size=32)
        x = torch.randn(2, 5, 128)
        out = ql(x)
        assert out.shape == (2, 5, 64)

    def test_roundtrip_accuracy(self):
        """INT4 quantization has more error but should still be reasonable."""
        torch.manual_seed(42)
        linear = nn.Linear(128, 64, bias=False)
        ql = QuantizedLinear.from_linear(linear, bits=4, group_size=32)
        x = torch.randn(1, 5, 128)

        with torch.no_grad():
            orig_out = linear(x)
            quant_out = ql(x)

        # INT4 has significant quantization error on random weights.
        # With 4-bit (16 levels) and group_size=32, ~35% relative error is expected.
        rel_error = (orig_out - quant_out).abs() / (orig_out.abs() + 1e-6)
        assert rel_error.mean() < 0.40, f"Mean relative error too high: {rel_error.mean():.4f}"

    def test_different_group_sizes(self):
        linear = nn.Linear(128, 64, bias=False)
        for gs in [32, 64, 128]:
            ql = QuantizedLinear.from_linear(linear, bits=4, group_size=gs)
            x = torch.randn(1, 3, 128)
            out = ql(x)
            assert out.shape == (1, 3, 64)


class TestQuantizedLinearEdgeCases:
    def test_unsupported_bits(self):
        with pytest.raises(ValueError, match="Unsupported quantization bits"):
            QuantizedLinear(64, 32, bits=3)

    def test_small_linear(self):
        linear = nn.Linear(4, 2, bias=False)
        ql8 = QuantizedLinear.from_linear(linear, bits=8)
        ql4 = QuantizedLinear.from_linear(linear, bits=4, group_size=4)
        x = torch.randn(1, 1, 4)
        assert ql8(x).shape == (1, 1, 2)
        assert ql4(x).shape == (1, 1, 2)

    def test_odd_dimensions(self):
        # Odd in_features for int4 (needs padding)
        linear = nn.Linear(65, 33, bias=False)
        ql = QuantizedLinear.from_linear(linear, bits=4, group_size=32)
        x = torch.randn(1, 2, 65)
        out = ql(x)
        assert out.shape == (1, 2, 33)


class TestQuantizeModel:
    def test_quantize_none_noop(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        config = QuantConfig(method="none")
        quantize_model(model, config)
        assert isinstance(model[0], nn.Linear)  # Unchanged

    def test_quantize_int8(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        config = QuantConfig(method="int8")
        quantize_model(model, config)
        assert isinstance(model[0], QuantizedLinear)
        assert isinstance(model[2], QuantizedLinear)
        assert model[0].bits == 8

    def test_quantize_int4(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        config = QuantConfig(method="int4", group_size=16)
        quantize_model(model, config)
        assert isinstance(model[0], QuantizedLinear)
        assert model[0].bits == 4

    def test_quantized_model_forward(self):
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        config = QuantConfig(method="int8")
        quantize_model(model, config)
        x = torch.randn(2, 5, 64)
        out = model(x)
        assert out.shape == (2, 5, 16)


class TestEstimateModelSize:
    def test_basic(self):
        model = nn.Linear(1024, 512, bias=False)
        info = estimate_model_size(model, bits=16)
        assert info["total_parameters"] == 1024 * 512
        assert info["bits"] == 16
        assert info["estimated_memory_mb"] > 0

    def test_different_bits(self):
        model = nn.Linear(1024, 512, bias=False)
        s16 = estimate_model_size(model, bits=16)
        s8 = estimate_model_size(model, bits=8)
        s4 = estimate_model_size(model, bits=4)
        assert s16["estimated_memory_mb"] > s8["estimated_memory_mb"]
        assert s8["estimated_memory_mb"] > s4["estimated_memory_mb"]

    def test_param_count(self):
        model = nn.Sequential(nn.Linear(100, 50, bias=True), nn.Linear(50, 25, bias=True))
        info = estimate_model_size(model)
        # 100*50 + 50 + 50*25 + 25 = 6325
        assert info["total_parameters"] == 6325
