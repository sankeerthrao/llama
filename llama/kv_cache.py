# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Advanced KV cache management for efficient inference.
Includes dynamic allocation, paged attention cache, and memory-efficient strategies.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class CacheConfig:
    """Configuration for KV cache behavior."""

    max_batch_size: int = 32
    max_seq_len: int = 2048
    n_heads: int = 32
    n_kv_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    page_size: int = 64  # Tokens per page for paged attention
    max_pages: int = 1024  # Maximum number of pages in the pool


class KVCache:
    """Standard contiguous KV cache with dynamic management.

    This cache pre-allocates a fixed-size tensor for keys and values,
    but tracks actual usage per sequence for efficient memory reporting.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.cache_k = torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.cache_v = torch.zeros(
            (max_batch_size, max_seq_len, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

        # Track current position per batch element
        self._positions = [0] * max_batch_size

    def update(
        self,
        batch_idx: int,
        start_pos: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the cache for a batch element and return the full cached keys/values.

        Args:
            batch_idx: Index in the batch.
            start_pos: Starting position in the sequence.
            keys: New key tensor [seq_len, n_kv_heads, head_dim].
            values: New value tensor [seq_len, n_kv_heads, head_dim].

        Returns:
            Tuple of (cached_keys, cached_values) up to start_pos + seq_len.
        """
        seq_len = keys.size(0)
        end_pos = start_pos + seq_len

        self.cache_k[batch_idx, start_pos:end_pos] = keys.to(self.dtype)
        self.cache_v[batch_idx, start_pos:end_pos] = values.to(self.dtype)
        self._positions[batch_idx] = end_pos

        return (
            self.cache_k[batch_idx, :end_pos],
            self.cache_v[batch_idx, :end_pos],
        )

    def get(self, batch_idx: int, end_pos: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys and values for a batch element.

        Args:
            batch_idx: Index in the batch.
            end_pos: End position (exclusive). If None, uses tracked position.

        Returns:
            Tuple of (cached_keys, cached_values).
        """
        if end_pos is None:
            end_pos = self._positions[batch_idx]
        return (
            self.cache_k[batch_idx, :end_pos],
            self.cache_v[batch_idx, :end_pos],
        )

    def reset(self, batch_idx: Optional[int] = None):
        """Reset cache for one or all batch elements.

        Args:
            batch_idx: Specific batch element to reset. If None, resets all.
        """
        if batch_idx is not None:
            self.cache_k[batch_idx].zero_()
            self.cache_v[batch_idx].zero_()
            self._positions[batch_idx] = 0
        else:
            self.cache_k.zero_()
            self.cache_v.zero_()
            self._positions = [0] * self.max_batch_size

    @property
    def memory_usage_bytes(self) -> int:
        """Total memory used by the cache in bytes."""
        element_size = self.cache_k.element_size()
        total_elements = self.cache_k.numel() + self.cache_v.numel()
        return total_elements * element_size

    @property
    def memory_usage_mb(self) -> float:
        """Total memory used by the cache in megabytes."""
        return self.memory_usage_bytes / (1024 * 1024)


class PagedKVCache:
    """Paged KV cache for memory-efficient variable-length sequence handling.

    Instead of pre-allocating a full [batch, max_seq_len, ...] tensor,
    this cache allocates fixed-size pages on demand. This reduces memory waste
    for batches with varying sequence lengths.
    """

    def __init__(
        self,
        n_kv_heads: int,
        head_dim: int,
        page_size: int = 64,
        max_pages: int = 1024,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages = max_pages
        self.device = device
        self.dtype = dtype

        # Page pool: pre-allocated pages
        self.k_pages = torch.zeros(
            (max_pages, page_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.v_pages = torch.zeros(
            (max_pages, page_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

        # Free page tracking
        self._free_pages: List[int] = list(range(max_pages))
        # Per-sequence page tables: sequence_id -> list of page indices
        self._page_tables: dict = {}
        # Per-sequence token counts
        self._seq_lengths: dict = {}

    @property
    def num_free_pages(self) -> int:
        """Number of available pages in the pool."""
        return len(self._free_pages)

    @property
    def num_allocated_pages(self) -> int:
        """Number of pages currently in use."""
        return self.max_pages - len(self._free_pages)

    def allocate_sequence(self, seq_id: int) -> bool:
        """Allocate cache space for a new sequence.

        Args:
            seq_id: Unique identifier for the sequence.

        Returns:
            True if allocation succeeded, False if seq_id already exists.
        """
        if seq_id in self._page_tables:
            return False
        self._page_tables[seq_id] = []
        self._seq_lengths[seq_id] = 0
        return True

    def _allocate_page(self) -> Optional[int]:
        """Allocate a single page from the pool.

        Returns:
            Page index, or None if no pages available.
        """
        if not self._free_pages:
            return None
        return self._free_pages.pop()

    def _free_page(self, page_idx: int):
        """Return a page to the free pool."""
        self.k_pages[page_idx].zero_()
        self.v_pages[page_idx].zero_()
        self._free_pages.append(page_idx)

    def append(
        self,
        seq_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> bool:
        """Append new key-value pairs to a sequence's cache.

        Args:
            seq_id: Sequence identifier.
            keys: New keys [seq_len, n_kv_heads, head_dim].
            values: New values [seq_len, n_kv_heads, head_dim].

        Returns:
            True if successful, False if out of pages.
        """
        if seq_id not in self._page_tables:
            self.allocate_sequence(seq_id)

        seq_len = keys.size(0)
        current_len = self._seq_lengths[seq_id]
        pages = self._page_tables[seq_id]

        for i in range(seq_len):
            pos_in_cache = current_len + i
            page_idx_in_table = pos_in_cache // self.page_size
            offset_in_page = pos_in_cache % self.page_size

            # Allocate new page if needed
            while page_idx_in_table >= len(pages):
                new_page = self._allocate_page()
                if new_page is None:
                    return False
                pages.append(new_page)

            physical_page = pages[page_idx_in_table]
            self.k_pages[physical_page, offset_in_page] = keys[i].to(self.dtype)
            self.v_pages[physical_page, offset_in_page] = values[i].to(self.dtype)

        self._seq_lengths[seq_id] = current_len + seq_len
        return True

    def get(self, seq_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get all cached keys and values for a sequence.

        Args:
            seq_id: Sequence identifier.

        Returns:
            Tuple of (keys, values) tensors, or None if sequence not found.
        """
        if seq_id not in self._page_tables:
            return None

        seq_len = self._seq_lengths[seq_id]
        if seq_len == 0:
            return (
                torch.zeros(0, self.n_kv_heads, self.head_dim, device=self.device, dtype=self.dtype),
                torch.zeros(0, self.n_kv_heads, self.head_dim, device=self.device, dtype=self.dtype),
            )

        pages = self._page_tables[seq_id]
        k_parts = []
        v_parts = []

        remaining = seq_len
        for page_idx in pages:
            count = min(remaining, self.page_size)
            k_parts.append(self.k_pages[page_idx, :count])
            v_parts.append(self.v_pages[page_idx, :count])
            remaining -= count
            if remaining <= 0:
                break

        return (torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0))

    def free_sequence(self, seq_id: int):
        """Free all pages allocated to a sequence.

        Args:
            seq_id: Sequence identifier.
        """
        if seq_id not in self._page_tables:
            return
        for page_idx in self._page_tables[seq_id]:
            self._free_page(page_idx)
        del self._page_tables[seq_id]
        del self._seq_lengths[seq_id]

    def free_all(self):
        """Free all sequences and reset the cache."""
        seq_ids = list(self._page_tables.keys())
        for seq_id in seq_ids:
            self.free_sequence(seq_id)

    @property
    def memory_usage_mb(self) -> float:
        """Approximate memory usage of allocated pages in MB."""
        allocated = self.num_allocated_pages
        page_bytes = self.page_size * self.n_kv_heads * self.head_dim * self.k_pages.element_size()
        return (allocated * page_bytes * 2) / (1024 * 1024)  # *2 for k and v

    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "total_pages": self.max_pages,
            "allocated_pages": self.num_allocated_pages,
            "free_pages": self.num_free_pages,
            "active_sequences": len(self._page_tables),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "page_size": self.page_size,
        }


class SlidingWindowCache:
    """Sliding window KV cache for bounded memory usage.

    Only retains the most recent `window_size` tokens for each sequence.
    Older tokens are evicted. Useful for long-context inference where
    full attention over the entire history is not needed.
    """

    def __init__(
        self,
        max_batch_size: int,
        window_size: int,
        n_kv_heads: int,
        head_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.max_batch_size = max_batch_size
        self.window_size = window_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.zeros(
            (max_batch_size, window_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.cache_v = torch.zeros(
            (max_batch_size, window_size, n_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._write_pos = [0] * max_batch_size
        self._total_written = [0] * max_batch_size

    def update(
        self,
        batch_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add new keys/values using circular buffer strategy.

        Note: After the buffer wraps around, the returned tensors contain
        entries in circular-buffer order, NOT chronological order. Since
        LLaMA uses Rotary Position Embeddings (RoPE) which encode absolute
        position into the key/value vectors, the physical ordering in memory
        does not affect attention correctness. Callers that need chronological
        order should reindex using the write position.

        Args:
            batch_idx: Batch index.
            keys: [seq_len, n_kv_heads, head_dim].
            values: [seq_len, n_kv_heads, head_dim].

        Returns:
            Tuple of all cached (keys, values) in the window.
        """
        seq_len = keys.size(0)
        for i in range(seq_len):
            pos = self._write_pos[batch_idx] % self.window_size
            self.cache_k[batch_idx, pos] = keys[i]
            self.cache_v[batch_idx, pos] = values[i]
            self._write_pos[batch_idx] += 1
            self._total_written[batch_idx] += 1

        valid_len = min(self._write_pos[batch_idx], self.window_size)
        return (
            self.cache_k[batch_idx, :valid_len],
            self.cache_v[batch_idx, :valid_len],
        )

    def reset(self, batch_idx: Optional[int] = None):
        """Reset cache for one or all batch elements."""
        if batch_idx is not None:
            self.cache_k[batch_idx].zero_()
            self.cache_v[batch_idx].zero_()
            self._write_pos[batch_idx] = 0
            self._total_written[batch_idx] = 0
        else:
            self.cache_k.zero_()
            self.cache_v.zero_()
            self._write_pos = [0] * self.max_batch_size
            self._total_written = [0] * self.max_batch_size
