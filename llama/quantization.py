# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Model quantization utilities for reduced memory usage and faster inference.
Supports INT8 and INT4 quantization with optional GPU acceleration.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class QuantConfig:
    """Quantization configuration."""

    method: str = "none"  # "none", "int8", "int4"
    group_size: int = 128


class QuantizedLinear(nn.Module):
    """A linear layer with quantized weights for reduced memory and faster inference.

    Stores weights in reduced precision and dequantizes on-the-fly during forward pass.
    Supports INT8 (per-channel) and INT4 (grouped) quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bits: int = 8,
        group_size: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        if bits == 8:
            self.register_buffer("weight_quantized", torch.zeros(out_features, in_features, dtype=torch.int8))
            self.register_buffer("weight_scale", torch.zeros(out_features, dtype=torch.float16))
        elif bits == 4:
            # Pack 2 int4 values per byte
            packed_size = math.ceil(in_features / 2)
            n_groups = math.ceil(in_features / group_size)
            self.register_buffer("weight_quantized", torch.zeros(out_features, packed_size, dtype=torch.uint8))
            self.register_buffer("weight_scale", torch.zeros(out_features, n_groups, dtype=torch.float16))
            self.register_buffer("weight_zero_point", torch.zeros(out_features, n_groups, dtype=torch.float16))
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def _dequantize_int8(self) -> torch.Tensor:
        """Dequantize INT8 weights to float."""
        return self.weight_quantized.float() * self.weight_scale.unsqueeze(1).float()

    def _dequantize_int4(self) -> torch.Tensor:
        """Dequantize INT4 weights to float."""
        # Unpack int4 from uint8 (unsigned [0, 15] range, matching quantization)
        high = (self.weight_quantized >> 4).to(torch.uint8)
        low = (self.weight_quantized & 0x0F).to(torch.uint8)
        # Interleave high and low nibbles
        unpacked = torch.stack([high, low], dim=-1).reshape(self.out_features, -1)
        unpacked = unpacked[:, : self.in_features]  # Trim padding

        # Apply per-group scale and zero point: original = q * scale + zero_point
        weight = unpacked.float()
        group_size = self.group_size
        n_groups = self.weight_scale.size(1)
        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, self.in_features)
            weight[:, start:end] = (
                weight[:, start:end] * self.weight_scale[:, g : g + 1].float()
                + self.weight_zero_point[:, g : g + 1].float()
            )
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        if self.bits == 8:
            weight = self._dequantize_int8()
        else:
            weight = self._dequantize_int4()

        weight = weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        output = torch.nn.functional.linear(x, weight, bias)
        return output

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 8, group_size: int = 128) -> "QuantizedLinear":
        """Create a quantized linear layer from a standard linear layer.

        Args:
            linear: Source linear layer with float weights.
            bits: Quantization bits (8 or 4).
            group_size: Group size for INT4 quantization.

        Returns:
            QuantizedLinear with quantized weights.
        """
        has_bias = linear.bias is not None
        ql = cls(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            bits=bits,
            group_size=group_size,
        )

        weight = linear.weight.data.float()

        if bits == 8:
            # Per-channel INT8 quantization
            scale = weight.abs().max(dim=1).values / 127.0
            scale = scale.clamp(min=1e-10)
            quantized = (weight / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
            ql.weight_quantized = quantized
            ql.weight_scale = scale.half()
        elif bits == 4:
            # Grouped INT4 quantization
            n_groups = math.ceil(linear.in_features / group_size)
            scales = []
            zero_points = []
            packed_rows = []

            for row in range(linear.out_features):
                row_scales = []
                row_zps = []
                quantized_row = []

                for g in range(n_groups):
                    start = g * group_size
                    end = min(start + group_size, linear.in_features)
                    group_weight = weight[row, start:end]

                    w_min = group_weight.min()
                    w_max = group_weight.max()
                    scale = (w_max - w_min) / 15.0
                    scale = scale.clamp(min=1e-10)
                    zero_point = w_min

                    q_vals = ((group_weight - zero_point) / scale).round().clamp(0, 15).to(torch.uint8)
                    quantized_row.append(q_vals)
                    row_scales.append(scale)
                    row_zps.append(zero_point)

                # Pack pairs of int4 into uint8
                full_row = torch.cat(quantized_row)
                if full_row.size(0) % 2 != 0:
                    full_row = torch.cat([full_row, torch.zeros(1, dtype=torch.uint8)])
                packed = (full_row[::2] << 4) | full_row[1::2]
                packed_rows.append(packed)
                scales.append(torch.tensor(row_scales))
                zero_points.append(torch.tensor(row_zps))

            ql.weight_quantized = torch.stack(packed_rows).to(torch.uint8)
            ql.weight_scale = torch.stack(scales).half()
            ql.weight_zero_point = torch.stack(zero_points).half()

        if has_bias:
            ql.bias = linear.bias.data.half()

        return ql


def _is_linear_layer(module: nn.Module) -> bool:
    """Check if a module is a linear layer (including fairscale parallel variants)."""
    if isinstance(module, nn.Linear):
        return True
    # Check for fairscale parallel linear layers
    try:
        from fairscale.nn.model_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )
        if isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
            return True
    except ImportError:
        pass
    return False


def _extract_linear_params(module: nn.Module) -> Optional[nn.Linear]:
    """Extract a plain nn.Linear equivalent from a module for quantization.

    For fairscale parallel layers, creates a temporary nn.Linear with the
    same weight and bias for quantization purposes.
    """
    if isinstance(module, nn.Linear):
        return module
    # Handle fairscale layers which store weight as nn.Parameter
    if hasattr(module, "weight"):
        weight = module.weight
        bias = getattr(module, "bias", None)
        linear = nn.Linear(weight.size(1), weight.size(0), bias=bias is not None)
        linear.weight = nn.Parameter(weight.data.clone())
        if bias is not None:
            linear.bias = nn.Parameter(bias.data.clone())
        return linear
    return None


def quantize_model(model: nn.Module, config: QuantConfig) -> nn.Module:
    """Quantize all linear layers in a model (including fairscale parallel layers).

    Args:
        model: The model to quantize.
        config: Quantization configuration.

    Returns:
        The quantized model (modified in-place).
    """
    if config.method == "none":
        return model

    bits = 8 if config.method == "int8" else 4

    replacements = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if _is_linear_layer(child):
                linear = _extract_linear_params(child)
                if linear is not None:
                    quantized = QuantizedLinear.from_linear(linear, bits=bits, group_size=config.group_size)
                    replacements.append((module, child_name, quantized))

    # Apply replacements after iteration to avoid modifying during traversal
    for parent, child_name, quantized in replacements:
        setattr(parent, child_name, quantized)

    return model


def estimate_model_size(model: nn.Module, bits: int = 16) -> dict:
    """Estimate model memory footprint.

    Args:
        model: The model to analyze.
        bits: Precision bits for estimation (16, 8, or 4).

    Returns:
        Dictionary with parameter count and estimated memory in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    bytes_per_param = bits / 8.0
    estimated_mb = (total_params * bytes_per_param) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "estimated_memory_mb": round(estimated_mb, 2),
        "bits": bits,
    }
