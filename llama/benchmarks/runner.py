# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Benchmarking suite for LLaMA inference engine.

Measures tokenization speed, model forward-pass latency, and sampling
strategy performance. All benchmarks work on CPU — CUDA is used only
when available and the model resides on GPU.
"""

from __future__ import annotations

import math
import time
import timeit
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration & result data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])
    seq_lengths: List[int] = field(default_factory=lambda: [32, 128, 512])
    num_iterations: int = 50
    warmup_iterations: int = 5


@dataclass
class BenchmarkResult:
    """Result of a single benchmark measurement."""

    name: str
    avg_latency: float  # seconds
    p50: float  # seconds
    p95: float  # seconds
    p99: float  # seconds
    throughput_tokens_per_sec: float
    memory_peak_mb: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Compute the *pct*-th percentile from a **sorted** list of floats."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100.0)
    floor_idx = int(math.floor(k))
    ceil_idx = int(math.ceil(k))
    if floor_idx == ceil_idx:
        return sorted_values[floor_idx]
    frac = k - floor_idx
    return sorted_values[floor_idx] * (1 - frac) + sorted_values[ceil_idx] * frac


def _peak_memory_mb() -> float:
    """Return peak GPU memory in MB if CUDA is available, else 0."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _make_result(
    name: str,
    latencies: List[float],
    total_tokens: int,
) -> BenchmarkResult:
    """Build a :class:`BenchmarkResult` from raw latency samples."""
    sorted_lat = sorted(latencies)
    total_time = sum(latencies)
    avg = total_time / len(latencies) if latencies else 0.0
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    return BenchmarkResult(
        name=name,
        avg_latency=avg,
        p50=_percentile(sorted_lat, 50),
        p95=_percentile(sorted_lat, 95),
        p99=_percentile(sorted_lat, 99),
        throughput_tokens_per_sec=throughput,
        memory_peak_mb=_peak_memory_mb(),
    )


# ---------------------------------------------------------------------------
# Tokenizer benchmark
# ---------------------------------------------------------------------------


class TokenizerBenchmark:
    """Measures tokenization encode/decode throughput."""

    def __init__(self, config: Optional[BenchmarkConfig] = None) -> None:
        self.config = config or BenchmarkConfig()

    def run(
        self,
        tokenizer: object,
        sample_texts: Optional[List[str]] = None,
    ) -> List[BenchmarkResult]:
        """Run encode/decode benchmarks using the given tokenizer.

        The tokenizer must expose ``encode(text, bos, eos)`` and
        ``decode(token_ids)`` methods (matching :class:`llama.Tokenizer`).

        Args:
            tokenizer: A tokenizer instance.
            sample_texts: Texts to benchmark. Defaults to synthetic samples.

        Returns:
            List of :class:`BenchmarkResult` — one for encode, one for decode.
        """
        if sample_texts is None:
            sample_texts = [
                "The quick brown fox jumps over the lazy dog. " * 10,
                "In a hole in the ground there lived a hobbit. " * 20,
                "To be or not to be, that is the question. " * 30,
            ]

        results: List[BenchmarkResult] = []

        # --- Encode benchmark ---
        encode_latencies: List[float] = []
        total_chars = 0
        for _ in range(self.config.warmup_iterations):
            for text in sample_texts:
                tokenizer.encode(text, bos=True, eos=False)  # type: ignore[union-attr]

        for _ in range(self.config.num_iterations):
            for text in sample_texts:
                start = time.perf_counter()
                tokenizer.encode(text, bos=True, eos=False)  # type: ignore[union-attr]
                elapsed = time.perf_counter() - start
                encode_latencies.append(elapsed)
                total_chars += len(text)

        results.append(_make_result("tokenizer_encode", encode_latencies, total_chars))

        # --- Decode benchmark ---
        # Pre-encode all texts to get token ID lists
        encoded_samples = [
            tokenizer.encode(t, bos=True, eos=False)  # type: ignore[union-attr]
            for t in sample_texts
        ]
        decode_latencies: List[float] = []
        total_tokens = 0

        for _ in range(self.config.warmup_iterations):
            for ids in encoded_samples:
                tokenizer.decode(ids)  # type: ignore[union-attr]

        for _ in range(self.config.num_iterations):
            for ids in encoded_samples:
                start = time.perf_counter()
                tokenizer.decode(ids)  # type: ignore[union-attr]
                elapsed = time.perf_counter() - start
                decode_latencies.append(elapsed)
                total_tokens += len(ids)

        results.append(_make_result("tokenizer_decode", decode_latencies, total_tokens))
        return results


# ---------------------------------------------------------------------------
# Model benchmark
# ---------------------------------------------------------------------------


class ModelBenchmark:
    """Measures forward-pass latency for different batch/sequence sizes.

    Automatically skips if the model requires CUDA and no GPU is available.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None) -> None:
        self.config = config or BenchmarkConfig()

    def run(
        self,
        model: torch.nn.Module,
        vocab_size: int = 32000,
    ) -> List[BenchmarkResult]:
        """Run forward-pass benchmarks on the given model.

        Args:
            model: A ``torch.nn.Module`` whose ``forward`` accepts a token
                   tensor and a ``start_pos`` int.
            vocab_size: Vocabulary size for random token generation.

        Returns:
            List of :class:`BenchmarkResult`, one per (batch, seq) combination.
        """
        device = next(model.parameters()).device

        # Skip GPU-only models when running on CPU
        if device.type == "cuda" and not torch.cuda.is_available():
            print("[ModelBenchmark] Skipping — model is on CUDA but no GPU available.")
            return []

        results: List[BenchmarkResult] = []

        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.seq_lengths:
                name = f"model_forward_bs{batch_size}_seq{seq_len}"
                tokens = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=device
                )

                # Warmup
                for _ in range(self.config.warmup_iterations):
                    with torch.no_grad():
                        try:
                            model(tokens, start_pos=0)
                        except Exception:
                            # Model may not support this combination; skip.
                            break

                # Timed iterations
                latencies: List[float] = []
                total_tokens = 0
                for _ in range(self.config.num_iterations):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.no_grad():
                        try:
                            model(tokens, start_pos=0)
                        except Exception:
                            break
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)
                    total_tokens += batch_size * seq_len

                if latencies:
                    results.append(_make_result(name, latencies, total_tokens))

        return results


# ---------------------------------------------------------------------------
# Sampling benchmark
# ---------------------------------------------------------------------------


class SamplingBenchmark:
    """Measures sampling-strategy performance using random logit tensors on CPU."""

    def __init__(self, config: Optional[BenchmarkConfig] = None) -> None:
        self.config = config or BenchmarkConfig()

    def run(self, vocab_size: int = 32000) -> List[BenchmarkResult]:
        """Run sampling benchmarks for various strategies.

        All computations happen on CPU with random logits tensors.

        Args:
            vocab_size: Vocabulary size for the logits tensor.

        Returns:
            List of :class:`BenchmarkResult`, one per sampling strategy.
        """
        batch_size = 1
        logits = torch.randn(batch_size, vocab_size)

        strategies = {
            "sampling_greedy": self._greedy,
            "sampling_top_k": self._top_k,
            "sampling_top_p": self._top_p,
            "sampling_temperature": self._temperature,
            "sampling_min_p": self._min_p,
        }

        results: List[BenchmarkResult] = []
        for name, fn in strategies.items():
            # Warmup
            for _ in range(self.config.warmup_iterations):
                fn(logits.clone())

            latencies: List[float] = []
            for _ in range(self.config.num_iterations):
                inp = logits.clone()
                start = time.perf_counter()
                fn(inp)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

            total_tokens = self.config.num_iterations * batch_size
            results.append(_make_result(name, latencies, total_tokens))

        return results

    # -- Strategy implementations (self-contained, no external deps) --

    @staticmethod
    def _greedy(logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True)

    @staticmethod
    def _top_k(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
        k = min(k, logits.size(-1))
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        threshold = top_k_values[:, -1].unsqueeze(-1)
        filtered = logits.masked_fill(logits < threshold, float("-inf"))
        probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def _top_p(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
        sorted_logits[to_remove] = float("-inf")
        probs = F.softmax(sorted_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        return torch.gather(sorted_indices, -1, sampled_idx)

    @staticmethod
    def _temperature(logits: torch.Tensor, temp: float = 0.8) -> torch.Tensor:
        scaled = logits / temp
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def _min_p(logits: torch.Tensor, min_p: float = 0.05) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        threshold = max_prob * min_p
        filtered = logits.masked_fill(probs < threshold, float("-inf"))
        filtered_probs = F.softmax(filtered, dim=-1)
        return torch.multinomial(filtered_probs, num_samples=1)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _print_results_table(results: List[BenchmarkResult]) -> None:
    """Pretty-print benchmark results as a formatted table."""
    if not results:
        print("No benchmark results to display.")
        return

    header = (
        f"{'Benchmark':<40} {'Avg (ms)':>10} {'P50 (ms)':>10} "
        f"{'P95 (ms)':>10} {'P99 (ms)':>10} {'Tok/s':>12} {'Mem (MB)':>10}"
    )
    separator = "-" * len(header)

    print()
    print(separator)
    print(header)
    print(separator)
    for r in results:
        print(
            f"{r.name:<40} "
            f"{r.avg_latency * 1000:>10.3f} "
            f"{r.p50 * 1000:>10.3f} "
            f"{r.p95 * 1000:>10.3f} "
            f"{r.p99 * 1000:>10.3f} "
            f"{r.throughput_tokens_per_sec:>12.1f} "
            f"{r.memory_peak_mb:>10.1f}"
        )
    print(separator)
    print()


def main() -> None:
    """Run all CPU-safe benchmarks and print a formatted results table."""
    print("=" * 60)
    print("  LLaMA Benchmark Suite")
    print("=" * 60)

    config = BenchmarkConfig(
        batch_sizes=[1, 4],
        seq_lengths=[32, 128],
        num_iterations=30,
        warmup_iterations=5,
    )

    all_results: List[BenchmarkResult] = []

    # --- Sampling benchmarks (always runnable on CPU) ---
    print("\n[*] Running sampling benchmarks...")
    sampling_bench = SamplingBenchmark(config=config)
    all_results.extend(sampling_bench.run())

    # --- Tokenizer benchmarks (only if a tokenizer model is available) ---
    # Skipped in main() since tokenizer requires a SentencePiece model file.
    # Use TokenizerBenchmark directly with a tokenizer instance to benchmark.
    print("[*] Tokenizer benchmark skipped (requires SentencePiece model file).")

    # --- Model benchmarks (only if model can be constructed on CPU) ---
    # The default Transformer requires fairscale parallel layers / CUDA.
    # Use ModelBenchmark directly with a model instance to benchmark.
    print("[*] Model benchmark skipped (requires model instance).")

    _print_results_table(all_results)


if __name__ == "__main__":
    main()
