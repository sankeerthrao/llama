# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""Tests for advanced sampling strategies."""

import pytest
import torch

from llama.sampling import (
    MirostatSampler,
    advanced_sample,
    apply_frequency_presence_penalty,
    apply_min_p,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_top_p,
)


class TestApplyTemperature:
    def test_temperature_1_noop(self, random_logits):
        result = apply_temperature(random_logits, 1.0)
        assert torch.equal(result, random_logits)

    def test_temperature_gt1_flattens(self, random_logits):
        result = apply_temperature(random_logits, 2.0)
        # Higher temperature should reduce the magnitude of logits
        assert result.abs().max() < random_logits.abs().max()

    def test_temperature_lt1_sharpens(self, random_logits):
        result = apply_temperature(random_logits, 0.5)
        # Lower temperature should increase the magnitude of logits
        assert result.abs().max() > random_logits.abs().max()

    def test_temperature_zero_raises(self, random_logits):
        with pytest.raises(ValueError, match="Temperature must be > 0"):
            apply_temperature(random_logits, 0.0)

    def test_temperature_negative_raises(self, random_logits):
        with pytest.raises(ValueError):
            apply_temperature(random_logits, -1.0)


class TestApplyTopK:
    def test_k_zero_noop(self, random_logits):
        result = apply_top_k(random_logits, 0)
        assert torch.equal(result, random_logits)

    def test_k_1_keeps_one(self, random_logits):
        result = apply_top_k(random_logits, 1)
        # Each row should have exactly 1 non-neg-inf value
        for row in result:
            finite_count = (row != float("-inf")).sum()
            assert finite_count == 1

    def test_k_10(self, random_logits):
        result = apply_top_k(random_logits, 10)
        for row in result:
            finite_count = (row != float("-inf")).sum()
            assert finite_count <= 10

    def test_k_larger_than_vocab(self, random_logits):
        result = apply_top_k(random_logits, 1000)
        # All values should be preserved (k > vocab_size)
        assert torch.equal(result, random_logits)

    def test_preserves_top_values(self, random_logits):
        k = 5
        result = apply_top_k(random_logits, k)
        for i in range(random_logits.size(0)):
            top_vals, _ = torch.topk(random_logits[i], k)
            result_finite = result[i][result[i] != float("-inf")]
            assert torch.allclose(result_finite.sort().values, top_vals.sort().values)


class TestApplyTopP:
    def test_p_1_noop(self, random_logits):
        result = apply_top_p(random_logits, 1.0)
        assert torch.equal(result, random_logits)

    def test_p_small_filters_most(self, random_logits):
        result = apply_top_p(random_logits, 0.1)
        for row in result:
            finite_count = (row != float("-inf")).sum()
            assert finite_count < random_logits.size(-1)
            assert finite_count >= 1  # At least one token kept

    def test_p_medium(self, random_logits):
        result = apply_top_p(random_logits, 0.5)
        # Should keep more tokens than p=0.1
        result_small = apply_top_p(random_logits, 0.1)
        for row_med, row_small in zip(result, result_small):
            count_med = (row_med != float("-inf")).sum()
            count_small = (row_small != float("-inf")).sum()
            assert count_med >= count_small


class TestApplyMinP:
    def test_disabled(self, random_logits):
        result = apply_min_p(random_logits, 0.0)
        assert torch.equal(result, random_logits)

    def test_filters_low_prob(self, random_logits):
        result = apply_min_p(random_logits, 0.5)
        for row in result:
            finite_count = (row != float("-inf")).sum()
            assert finite_count >= 1
            assert finite_count < random_logits.size(-1)

    def test_high_threshold_keeps_few(self, random_logits):
        result = apply_min_p(random_logits, 0.9)
        for row in result:
            finite_count = (row != float("-inf")).sum()
            assert finite_count >= 1


class TestRepetitionPenalty:
    def test_no_penalty(self, random_logits, sample_tokens):
        result = apply_repetition_penalty(random_logits, sample_tokens, 1.0)
        assert torch.equal(result, random_logits)

    def test_penalty_reduces_repeated(self, random_logits, sample_tokens):
        result = apply_repetition_penalty(random_logits, sample_tokens, 1.5)
        # Tokens that appeared should have lower logits (if positive)
        for i in range(random_logits.size(0)):
            for token_id in sample_tokens[i].unique():
                orig = random_logits[i, token_id].item()
                penalized = result[i, token_id].item()
                if orig > 0:
                    assert penalized < orig
                elif orig < 0:
                    assert penalized < orig  # More negative

    def test_does_not_modify_input(self, random_logits, sample_tokens):
        original = random_logits.clone()
        apply_repetition_penalty(random_logits, sample_tokens, 1.5)
        assert torch.equal(random_logits, original)


class TestFrequencyPresencePenalty:
    def test_no_penalty(self, random_logits, sample_tokens):
        result = apply_frequency_presence_penalty(random_logits, sample_tokens, 0.0, 0.0)
        assert torch.equal(result, random_logits)

    def test_frequency_penalty(self, random_logits, sample_tokens):
        result = apply_frequency_presence_penalty(random_logits, sample_tokens, 0.5, 0.0)
        # Repeated tokens should have lower logits
        assert not torch.equal(result, random_logits)
        # Token 5 appears 3 times in first sequence, should be penalized more
        assert result[0, 5] < random_logits[0, 5]

    def test_presence_penalty(self, random_logits, sample_tokens):
        result = apply_frequency_presence_penalty(random_logits, sample_tokens, 0.0, 0.5)
        assert not torch.equal(result, random_logits)

    def test_combined_penalties(self, random_logits, sample_tokens):
        result = apply_frequency_presence_penalty(random_logits, sample_tokens, 0.3, 0.3)
        assert not torch.equal(result, random_logits)


class TestMirostatSampler:
    def test_basic_sampling(self):
        torch.manual_seed(42)
        sampler = MirostatSampler(tau=3.0, eta=0.1)
        logits = torch.randn(100)
        token = sampler.sample(logits)
        assert token.numel() == 1
        assert 0 <= token.item() < 100

    def test_deterministic_with_seed(self):
        logits = torch.randn(100)
        torch.manual_seed(42)
        s1 = MirostatSampler(tau=3.0)
        t1 = s1.sample(logits)
        torch.manual_seed(42)
        s2 = MirostatSampler(tau=3.0)
        t2 = s2.sample(logits)
        assert t1.item() == t2.item()

    def test_reset(self):
        sampler = MirostatSampler()
        sampler.mu = 5.0
        sampler.reset()
        assert sampler.mu is None

    def test_low_tau_less_random(self):
        """Lower tau should produce less diverse tokens across many samples."""
        torch.manual_seed(42)
        logits = torch.randn(100)
        low_sampler = MirostatSampler(tau=0.5)
        high_sampler = MirostatSampler(tau=10.0)

        low_tokens = set()
        high_tokens = set()
        for _ in range(50):
            low_tokens.add(low_sampler.sample(logits.clone()).item())
            high_tokens.add(high_sampler.sample(logits.clone()).item())

        # High tau should generally produce more diverse tokens
        # (not always guaranteed due to randomness, but highly likely)
        assert len(high_tokens) >= len(low_tokens)


class TestSampleTopP:
    """Tests for the original Meta top-p sampling function."""

    def test_basic(self):
        torch.manual_seed(42)
        probs = torch.softmax(torch.randn(2, 100), dim=-1)
        result = sample_top_p(probs, 0.9)
        assert result.shape == (2, 1)
        assert (result >= 0).all()
        assert (result < 100).all()

    def test_p_1_samples_from_full(self):
        torch.manual_seed(42)
        probs = torch.softmax(torch.randn(1, 10), dim=-1)
        result = sample_top_p(probs, 1.0)
        assert 0 <= result.item() < 10


class TestAdvancedSample:
    def test_greedy(self, random_logits):
        result = advanced_sample(random_logits, temperature=0)
        expected = torch.argmax(random_logits, dim=-1, keepdim=True)
        assert torch.equal(result, expected)

    def test_returns_valid_indices(self, random_logits):
        torch.manual_seed(42)
        result = advanced_sample(random_logits, temperature=0.8, top_p=0.9)
        assert result.shape == (2, 1)
        assert (result >= 0).all()
        assert (result < 100).all()

    def test_with_top_k(self, random_logits):
        torch.manual_seed(42)
        result = advanced_sample(random_logits, temperature=0.8, top_k=5)
        assert result.shape == (2, 1)
        assert (result >= 0).all()
        assert (result < 100).all()

    def test_with_min_p(self, random_logits):
        torch.manual_seed(42)
        result = advanced_sample(random_logits, temperature=0.8, min_p=0.1)
        assert result.shape == (2, 1)

    def test_with_repetition_penalty(self, random_logits, sample_tokens):
        torch.manual_seed(42)
        result = advanced_sample(
            random_logits,
            temperature=0.8,
            repetition_penalty=1.5,
            generated_tokens=sample_tokens,
        )
        assert result.shape == (2, 1)

    def test_with_all_options(self, random_logits, sample_tokens):
        torch.manual_seed(42)
        result = advanced_sample(
            random_logits,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            min_p=0.05,
            repetition_penalty=1.2,
            frequency_penalty=0.3,
            presence_penalty=0.3,
            generated_tokens=sample_tokens,
        )
        assert result.shape == (2, 1)
        assert (result >= 0).all()
        assert (result < 100).all()

    def test_large_vocab(self, random_logits_large):
        torch.manual_seed(42)
        result = advanced_sample(random_logits_large, temperature=0.6, top_p=0.9, top_k=40)
        assert result.shape == (4, 1)
        assert (result >= 0).all()
        assert (result < 32000).all()
