# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""Shared fixtures for the LLaMA test suite."""

import pytest
import torch


@pytest.fixture
def device():
    """Return the appropriate device for testing."""
    return "cpu"


@pytest.fixture
def random_logits():
    """Generate random logits for sampling tests."""
    torch.manual_seed(42)
    return torch.randn(2, 100)  # batch_size=2, vocab_size=100


@pytest.fixture
def random_logits_large():
    """Generate larger random logits for performance-sensitive tests."""
    torch.manual_seed(42)
    return torch.randn(4, 32000)  # batch_size=4, vocab_size=32000


@pytest.fixture
def sample_tokens():
    """Generate sample token sequences for penalty tests."""
    return torch.tensor([
        [1, 5, 3, 5, 5, 8, 1, 3],
        [2, 7, 7, 7, 4, 4, 9, 0],
    ])
