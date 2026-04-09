# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
LLaMA Inference Engine - High-performance inference for LLaMA models.

Core API:
    Llama, Dialog - Chat and text generation (original Meta API)
    ModelArgs, Transformer - Model architecture
    Tokenizer - Text tokenization

Enhanced Features:
    InferenceConfig - Comprehensive configuration system
    CPUGenerator, CPUModelLoader - CPU-only inference support
    QuantizedLinear, quantize_model - INT8/INT4 quantization
    KVCache, PagedKVCache, SlidingWindowCache - Advanced KV cache management
    advanced_sample - Flexible sampling with top-k/p, min-p, repetition penalty
"""

# These always work (no fairscale dependency)
from .config import InferenceConfig, ModelConfig, GenerationConfig, ServerConfig
from .sampling import advanced_sample, sample_top_p, MirostatSampler
from .quantization import QuantizedLinear, quantize_model, estimate_model_size, QuantConfig
from .kv_cache import KVCache, PagedKVCache, SlidingWindowCache
from .cpu_inference import CPUGenerator, CPUModelLoader, get_optimal_device, get_device_info

# These require fairscale -- import with fallback
try:
    from .generation import Llama, Dialog
    from .model import ModelArgs, Transformer
    from .tokenizer import Tokenizer
except ImportError:
    Llama = None  # type: ignore
    Dialog = None  # type: ignore
    ModelArgs = None  # type: ignore
    Transformer = None  # type: ignore
    Tokenizer = None  # type: ignore

__version__ = "2.0.0"

__all__ = [
    # Original API (requires fairscale)
    "Llama",
    "Dialog",
    "ModelArgs",
    "Transformer",
    "Tokenizer",
    # Configuration
    "InferenceConfig",
    "ModelConfig",
    "GenerationConfig",
    "ServerConfig",
    # Sampling
    "advanced_sample",
    "sample_top_p",
    "MirostatSampler",
    # CPU Inference
    "CPUGenerator",
    "CPUModelLoader",
    "get_optimal_device",
    "get_device_info",
    # Quantization
    "QuantizedLinear",
    "quantize_model",
    "estimate_model_size",
    "QuantConfig",
    # KV Cache
    "KVCache",
    "PagedKVCache",
    "SlidingWindowCache",
]
