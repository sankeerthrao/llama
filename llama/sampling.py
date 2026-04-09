# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Advanced sampling strategies for text generation.
Includes top-k, top-p (nucleus), min-p, temperature, repetition penalty,
frequency penalty, presence penalty, and mirostat sampling.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Temperature=0 is not valid here -- use greedy decoding (argmax) externally.
    Temperature=1.0 returns logits unchanged (no-op).

    Args:
        logits: Raw logits from the model [batch_size, vocab_size].
        temperature: Temperature value. Must be > 0. Higher = more random, lower = more deterministic.

    Returns:
        Temperature-scaled logits.

    Raises:
        ValueError: If temperature <= 0.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be > 0 for scaling. Use argmax for greedy decoding.")
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits.

    Args:
        logits: Logits tensor [batch_size, vocab_size].
        k: Number of top tokens to keep. 0 = disabled.

    Returns:
        Filtered logits with non-top-k values set to -inf.
    """
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits.

    Args:
        logits: Logits tensor [batch_size, vocab_size].
        p: Cumulative probability threshold. 1.0 = disabled.

    Returns:
        Filtered logits.
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs - sorted_probs > p
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Scatter filtered sorted values back to original indexing using a fresh output tensor
    output = torch.full_like(logits, float("-inf"))
    return output.scatter(1, sorted_indices, sorted_logits)


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Apply min-p filtering to logits.

    Filters out tokens whose probability is less than min_p times the
    probability of the most likely token.

    Args:
        logits: Logits tensor [batch_size, vocab_size].
        min_p: Minimum probability ratio threshold. 0 = disabled.

    Returns:
        Filtered logits.
    """
    if min_p <= 0:
        return logits
    probs = F.softmax(logits, dim=-1)
    max_probs = probs.max(dim=-1, keepdim=True).values
    threshold = max_probs * min_p
    return logits.masked_fill(probs < threshold, float("-inf"))


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits for previously generated tokens.

    Args:
        logits: Logits tensor [batch_size, vocab_size].
        generated_tokens: Previously generated token IDs [batch_size, seq_len].
        penalty: Repetition penalty factor. 1.0 = no penalty.

    Returns:
        Penalized logits.
    """
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    for i in range(logits.size(0)):
        unique_tokens = generated_tokens[i].unique()
        score = torch.gather(logits[i], 0, unique_tokens)
        # If score < 0 then repetition penalty increases the magnitude (more negative)
        # If score > 0 then repetition penalty decreases the magnitude (less positive)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits[i].scatter_(0, unique_tokens, score)

    return logits


def apply_frequency_presence_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """Apply frequency and presence penalties (OpenAI-style).

    Args:
        logits: Logits tensor [batch_size, vocab_size].
        generated_tokens: Previously generated token IDs [batch_size, seq_len].
        frequency_penalty: Penalty proportional to frequency of token. 0 = disabled.
        presence_penalty: Flat penalty for any token that appeared. 0 = disabled.

    Returns:
        Penalized logits.
    """
    if frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits

    logits = logits.clone()
    vocab_size = logits.size(-1)
    for i in range(logits.size(0)):
        token_ids = generated_tokens[i]
        # Filter out padding (negative values)
        valid_ids = token_ids[token_ids >= 0]
        # Vectorized frequency counting
        freq_counts = torch.bincount(valid_ids.long(), minlength=vocab_size).float().to(logits.device)
        freq_counts = freq_counts[:vocab_size]  # Trim if token ids exceed vocab

        presence_mask = (freq_counts > 0).float()
        logits[i] -= frequency_penalty * freq_counts
        logits[i] -= presence_penalty * presence_mask

    return logits


class MirostatSampler:
    """Mirostat v2 adaptive sampling.

    Maintains a target surprise level (tau) and adjusts the sampling
    distribution dynamically to achieve consistent perplexity.
    """

    def __init__(self, tau: float = 3.0, eta: float = 0.1):
        """
        Args:
            tau: Target surprise value (in bits). Higher = more random.
            eta: Learning rate for adapting mu.
        """
        self.tau = tau
        self.eta = eta
        self.mu: Optional[float] = None

    def reset(self):
        """Reset the sampler state."""
        self.mu = None

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample using Mirostat v2.

        Args:
            logits: Logits tensor [vocab_size] (single sample, not batched).

        Returns:
            Sampled token index.
        """
        if self.mu is None:
            self.mu = 2.0 * self.tau

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # Compute surprise values: -log2(p)
        surprise_values = -torch.log2(sorted_probs + 1e-10)

        # Find the truncation point
        # Keep tokens until we exceed mu
        cumulative_surprise = 0.0
        k = 0
        for idx in range(len(sorted_probs)):
            if surprise_values[idx] > self.mu:
                break
            k = idx + 1

        k = max(k, 1)  # Always keep at least 1 token

        # Truncate and renormalize
        truncated_probs = sorted_probs[:k]
        truncated_probs = truncated_probs / truncated_probs.sum()

        # Sample from truncated distribution
        sampled_idx = torch.multinomial(truncated_probs, num_samples=1)
        token = sorted_indices[sampled_idx]

        # Update mu
        observed_surprise = surprise_values[sampled_idx].item()
        self.mu = self.mu - self.eta * (observed_surprise - self.tau)

        return token


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Perform top-p (nucleus) sampling on a probability distribution.

    This is the original Meta implementation kept for backward compatibility.

    Args:
        probs: Probability distribution tensor.
        p: Probability threshold for top-p sampling.

    Returns:
        Sampled token indices.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def advanced_sample(
    logits: torch.Tensor,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    generated_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply the full sampling pipeline and return sampled tokens.

    Args:
        logits: Raw logits from the model [batch_size, vocab_size].
        temperature: Temperature scaling factor.
        top_p: Top-p (nucleus) sampling threshold.
        top_k: Top-k sampling threshold.
        min_p: Min-p sampling threshold.
        repetition_penalty: Repetition penalty factor.
        frequency_penalty: Frequency penalty (OpenAI-style).
        presence_penalty: Presence penalty (OpenAI-style).
        generated_tokens: Previously generated tokens for penalty computation.

    Returns:
        Sampled token indices [batch_size, 1].
    """
    # Apply repetition-based penalties first (operate on raw logits)
    if generated_tokens is not None and generated_tokens.numel() > 0:
        logits = apply_repetition_penalty(logits, generated_tokens, repetition_penalty)
        logits = apply_frequency_presence_penalty(
            logits, generated_tokens, frequency_penalty, presence_penalty
        )

    # Greedy decoding (before temperature scaling, which requires temperature > 0)
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # Temperature scaling (temperature > 0 guaranteed here)
    logits = apply_temperature(logits, temperature)

    # Apply filtering (order matters: top-k first, then top-p, then min-p)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    logits = apply_min_p(logits, min_p)

    # Convert to probabilities and sample
    probs = F.softmax(logits, dim=-1)

    # Handle edge case where all probs are 0 after filtering
    all_zero = probs.sum(dim=-1) == 0
    if all_zero.any():
        # Fall back to uniform distribution for affected samples
        probs[all_zero] = 1.0 / probs.size(-1)

    return torch.multinomial(probs, num_samples=1)
