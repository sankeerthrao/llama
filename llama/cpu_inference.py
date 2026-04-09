# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
CPU-only inference engine for LLaMA models.
Enables running models on machines without CUDA GPUs, with automatic
device selection, torch.compile optimization, and memory-efficient strategies.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CPUInferenceConfig:
    """Configuration for CPU inference."""

    num_threads: int = 0  # 0 = auto-detect
    use_torch_compile: bool = False  # torch.compile for PyTorch 2.0+
    dtype: Literal["float32", "bfloat16"] = "float32"
    enable_memory_efficient: bool = True


def get_optimal_device() -> str:
    """Detect the best available device for inference.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_info() -> dict:
    """Get detailed information about available compute devices.

    Returns:
        Dictionary with device capabilities and memory info.
    """
    info = {
        "selected_device": get_optimal_device(),
        "cuda_available": torch.cuda.is_available(),
        "cpu_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total_mb"] = round(torch.cuda.get_device_properties(0).total_mem / 1024 / 1024, 2)

    if hasattr(torch.backends, "mps"):
        info["mps_available"] = torch.backends.mps.is_available()

    return info


class CPUModelLoader:
    """Load and prepare LLaMA models for CPU inference.

    Handles device mapping, dtype conversion, and optional optimizations
    like torch.compile for CPU inference scenarios.
    """

    def __init__(self, config: Optional[CPUInferenceConfig] = None):
        self.config = config or CPUInferenceConfig()
        self._setup_threads()

    def _setup_threads(self):
        """Configure CPU thread count for optimal performance."""
        if self.config.num_threads > 0:
            torch.set_num_threads(self.config.num_threads)
        else:
            # Use all available cores
            cpu_count = os.cpu_count() or 4
            torch.set_num_threads(cpu_count)

    def _get_dtype(self) -> torch.dtype:
        """Get the torch dtype from config string."""
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.dtype, torch.float32)

    def load_model(
        self,
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int = 512,
        max_batch_size: int = 4,
    ) -> Tuple["Transformer", "Tokenizer"]:
        """Load a LLaMA model for CPU inference.

        Args:
            ckpt_dir: Path to checkpoint directory.
            tokenizer_path: Path to tokenizer model file.
            max_seq_len: Maximum sequence length.
            max_batch_size: Maximum batch size.

        Returns:
            Tuple of (model, tokenizer).
        """
        from llama.model import ModelArgs, Transformer
        from llama.tokenizer import Tokenizer

        start_time = time.time()

        # Load params
        params_path = Path(ckpt_dir) / "params.json"
        with open(params_path) as f:
            params = json.loads(f.read())

        model_args = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words

        # Build model on CPU with appropriate dtype
        target_dtype = self._get_dtype()
        with torch.device("cpu"):
            torch.set_default_dtype(target_dtype)
            model = Transformer(model_args)
            torch.set_default_dtype(torch.float32)

        # Load checkpoint(s)
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        if checkpoints:
            # For CPU inference, merge all shards
            merged_state = {}
            for ckpt_path in checkpoints:
                shard = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                merged_state.update(shard)
            model.load_state_dict(merged_state, strict=False)

        model = model.to(target_dtype)
        model.eval()

        # Apply torch.compile if available and enabled
        if self.config.use_torch_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass  # Silently fall back if compile fails

        elapsed = time.time() - start_time
        print(f"Loaded model on CPU in {elapsed:.2f} seconds")

        return model, tokenizer


class CPUGenerator:
    """Text generation engine for CPU-based inference.

    Provides the same API as the GPU-based Llama class but optimized for CPU,
    with automatic memory management and optional optimizations.
    """

    def __init__(self, model: "Transformer", tokenizer: "Tokenizer"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cpu"

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """Generate text from prompt tokens on CPU.

        Args:
            prompt_tokens: List of tokenized prompts.
            max_gen_len: Maximum generation length.
            temperature: Sampling temperature.
            top_p: Top-p sampling threshold.

        Returns:
            Tuple of (generated_tokens, None).
        """
        from llama.sampling import sample_top_p

        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=self.device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=self.device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        input_text_mask = tokens != pad_id

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            toks = toks[: len(prompt_tokens[i]) + max_gen_len]
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
            out_tokens.append(toks)

        return (out_tokens, None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ) -> List[dict]:
        """Generate text completions.

        Args:
            prompts: List of text prompts.
            temperature: Sampling temperature.
            top_p: Top-p threshold.
            max_gen_len: Max generation length.

        Returns:
            List of dicts with 'generation' key.
        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, _ = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]
