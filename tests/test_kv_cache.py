# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""Tests for KV cache implementations."""

import pytest
import torch

from llama.kv_cache import KVCache, PagedKVCache, SlidingWindowCache


class TestKVCache:
    @pytest.fixture
    def cache(self):
        return KVCache(
            max_batch_size=4,
            max_seq_len=64,
            n_kv_heads=4,
            head_dim=16,
            device="cpu",
            dtype=torch.float32,
        )

    def test_init(self, cache):
        assert cache.cache_k.shape == (4, 64, 4, 16)
        assert cache.cache_v.shape == (4, 64, 4, 16)
        assert cache.memory_usage_bytes > 0

    def test_update_and_get(self, cache):
        keys = torch.randn(5, 4, 16)
        values = torch.randn(5, 4, 16)
        k_out, v_out = cache.update(0, 0, keys, values)
        assert k_out.shape == (5, 4, 16)
        assert v_out.shape == (5, 4, 16)
        assert torch.allclose(k_out, keys)
        assert torch.allclose(v_out, values)

    def test_update_incremental(self, cache):
        k1 = torch.randn(3, 4, 16)
        v1 = torch.randn(3, 4, 16)
        cache.update(0, 0, k1, v1)

        k2 = torch.randn(2, 4, 16)
        v2 = torch.randn(2, 4, 16)
        k_out, v_out = cache.update(0, 3, k2, v2)
        assert k_out.shape == (5, 4, 16)
        assert v_out.shape == (5, 4, 16)
        assert torch.allclose(k_out[:3], k1)
        assert torch.allclose(k_out[3:], k2)

    def test_get(self, cache):
        keys = torch.randn(5, 4, 16)
        values = torch.randn(5, 4, 16)
        cache.update(0, 0, keys, values)
        k, v = cache.get(0)
        assert k.shape == (5, 4, 16)

    def test_get_with_end_pos(self, cache):
        keys = torch.randn(5, 4, 16)
        values = torch.randn(5, 4, 16)
        cache.update(0, 0, keys, values)
        k, v = cache.get(0, end_pos=3)
        assert k.shape == (3, 4, 16)

    def test_reset_single(self, cache):
        keys = torch.randn(5, 4, 16)
        values = torch.randn(5, 4, 16)
        cache.update(0, 0, keys, values)
        cache.update(1, 0, keys, values)
        cache.reset(0)
        k, v = cache.get(0)
        assert k.shape == (0, 4, 16)
        # Batch 1 should be unaffected
        k1, v1 = cache.get(1)
        assert k1.shape == (5, 4, 16)

    def test_reset_all(self, cache):
        keys = torch.randn(5, 4, 16)
        values = torch.randn(5, 4, 16)
        cache.update(0, 0, keys, values)
        cache.update(1, 0, keys, values)
        cache.reset()
        assert cache._positions == [0, 0, 0, 0]

    def test_memory_usage(self, cache):
        mb = cache.memory_usage_mb
        assert mb > 0
        # 2 * (4 * 64 * 4 * 16) * 4 bytes (float32) / 1024^2
        expected = 2 * 4 * 64 * 4 * 16 * 4 / (1024 * 1024)
        assert abs(mb - expected) < 0.01

    def test_multiple_batches_independent(self, cache):
        k0 = torch.ones(3, 4, 16)
        v0 = torch.ones(3, 4, 16)
        k1 = torch.zeros(3, 4, 16)
        v1 = torch.zeros(3, 4, 16)
        cache.update(0, 0, k0, v0)
        cache.update(1, 0, k1, v1)
        ko0, vo0 = cache.get(0)
        ko1, vo1 = cache.get(1)
        assert torch.allclose(ko0, k0)
        assert torch.allclose(ko1, k1)


class TestPagedKVCache:
    @pytest.fixture
    def cache(self):
        return PagedKVCache(
            n_kv_heads=4,
            head_dim=16,
            page_size=8,
            max_pages=32,
            device="cpu",
            dtype=torch.float32,
        )

    def test_init(self, cache):
        assert cache.num_free_pages == 32
        assert cache.num_allocated_pages == 0

    def test_allocate_sequence(self, cache):
        assert cache.allocate_sequence(0) is True
        assert cache.allocate_sequence(0) is False  # Already exists

    def test_append_and_get(self, cache):
        cache.allocate_sequence(0)
        keys = torch.randn(5, 4, 16)
        values = torch.randn(5, 4, 16)
        success = cache.append(0, keys, values)
        assert success is True

        k, v = cache.get(0)
        assert k.shape == (5, 4, 16)
        assert torch.allclose(k, keys)
        assert torch.allclose(v, values)

    def test_append_auto_allocates(self, cache):
        # Don't explicitly allocate -- append should auto-allocate
        keys = torch.randn(3, 4, 16)
        values = torch.randn(3, 4, 16)
        success = cache.append(99, keys, values)
        assert success is True
        k, v = cache.get(99)
        assert k.shape == (3, 4, 16)

    def test_append_incremental(self, cache):
        k1 = torch.randn(5, 4, 16)
        v1 = torch.randn(5, 4, 16)
        cache.append(0, k1, v1)

        k2 = torch.randn(3, 4, 16)
        v2 = torch.randn(3, 4, 16)
        cache.append(0, k2, v2)

        k, v = cache.get(0)
        assert k.shape == (8, 4, 16)
        assert torch.allclose(k[:5], k1)
        assert torch.allclose(k[5:], k2)

    def test_page_allocation(self, cache):
        # Page size is 8, so 10 tokens should need 2 pages
        keys = torch.randn(10, 4, 16)
        values = torch.randn(10, 4, 16)
        cache.append(0, keys, values)
        assert cache.num_allocated_pages == 2
        assert cache.num_free_pages == 30

    def test_free_sequence(self, cache):
        keys = torch.randn(10, 4, 16)
        values = torch.randn(10, 4, 16)
        cache.append(0, keys, values)
        assert cache.num_allocated_pages == 2
        cache.free_sequence(0)
        assert cache.num_allocated_pages == 0
        assert cache.num_free_pages == 32
        assert cache.get(0) is None

    def test_free_all(self, cache):
        for i in range(4):
            cache.append(i, torch.randn(5, 4, 16), torch.randn(5, 4, 16))
        assert cache.num_allocated_pages > 0
        cache.free_all()
        assert cache.num_allocated_pages == 0

    def test_out_of_pages(self):
        cache = PagedKVCache(
            n_kv_heads=4, head_dim=16, page_size=4, max_pages=2, device="cpu"
        )
        # 2 pages * 4 tokens = 8 tokens max
        success = cache.append(0, torch.randn(8, 4, 16), torch.randn(8, 4, 16))
        assert success is True
        # This should fail -- no more pages
        success = cache.append(1, torch.randn(1, 4, 16), torch.randn(1, 4, 16))
        assert success is False

    def test_get_empty_sequence(self, cache):
        cache.allocate_sequence(0)
        k, v = cache.get(0)
        assert k.shape == (0, 4, 16)
        assert v.shape == (0, 4, 16)

    def test_get_nonexistent(self, cache):
        assert cache.get(999) is None

    def test_stats(self, cache):
        cache.append(0, torch.randn(10, 4, 16), torch.randn(10, 4, 16))
        stats = cache.stats()
        assert stats["total_pages"] == 32
        assert stats["allocated_pages"] == 2
        assert stats["free_pages"] == 30
        assert stats["active_sequences"] == 1
        assert stats["page_size"] == 8

    def test_memory_usage(self, cache):
        assert cache.memory_usage_mb == 0.0
        cache.append(0, torch.randn(10, 4, 16), torch.randn(10, 4, 16))
        assert cache.memory_usage_mb > 0

    def test_free_nonexistent(self, cache):
        # Should not raise
        cache.free_sequence(999)


class TestSlidingWindowCache:
    @pytest.fixture
    def cache(self):
        return SlidingWindowCache(
            max_batch_size=2,
            window_size=8,
            n_kv_heads=4,
            head_dim=16,
            device="cpu",
            dtype=torch.float32,
        )

    def test_init(self, cache):
        assert cache.cache_k.shape == (2, 8, 4, 16)
        assert cache.window_size == 8

    def test_update_within_window(self, cache):
        keys = torch.randn(5, 4, 16)
        values = torch.randn(5, 4, 16)
        k_out, v_out = cache.update(0, keys, values)
        assert k_out.shape == (5, 4, 16)

    def test_update_exceeds_window(self, cache):
        # Write 12 tokens with window_size=8 -> should keep last 8
        k1 = torch.randn(12, 4, 16)
        v1 = torch.randn(12, 4, 16)
        k_out, v_out = cache.update(0, k1, v1)
        assert k_out.shape == (8, 4, 16)

    def test_update_incremental_overflow(self, cache):
        k1 = torch.randn(6, 4, 16)
        v1 = torch.randn(6, 4, 16)
        cache.update(0, k1, v1)

        k2 = torch.randn(4, 4, 16)
        v2 = torch.randn(4, 4, 16)
        k_out, v_out = cache.update(0, k2, v2)
        # 6 + 4 = 10 > 8, so window should be full
        assert k_out.shape == (8, 4, 16)

    def test_reset_single(self, cache):
        cache.update(0, torch.randn(5, 4, 16), torch.randn(5, 4, 16))
        cache.update(1, torch.randn(3, 4, 16), torch.randn(3, 4, 16))
        cache.reset(0)
        assert cache._write_pos[0] == 0
        assert cache._write_pos[1] == 3  # Unaffected

    def test_reset_all(self, cache):
        cache.update(0, torch.randn(5, 4, 16), torch.randn(5, 4, 16))
        cache.update(1, torch.randn(3, 4, 16), torch.randn(3, 4, 16))
        cache.reset()
        assert cache._write_pos == [0, 0]
        assert cache._total_written == [0, 0]
