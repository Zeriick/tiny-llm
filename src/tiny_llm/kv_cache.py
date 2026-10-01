from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import mlx.core as mx

from .attention import causal_mask

if TYPE_CHECKING:
    from .paged_kv_cache import PagedKvMetadata


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            The updated keys, updated values, sequence length, and mask. On
            On Week 2 Day 1, the mask is passed through unchanged. Week 3 Day 1
            uses the sequence length and mask to construct a dense batch.
        """

    def release(self):
        pass

    def materialize(self):
        """Evaluate owned K/V storage without changing its logical layout."""
        pass

    def update_and_fetch_paged(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        pass

    def rewind(self, n: int):
        pass


# S = max(S_i across active requests)
# L = mask_length (input parameter)
# request_keys: H, S_i, D
# request_values: H, S_i, D
# batched_keys: B, H, S, D
# batched_values: B, H, S, D
# mask: B, 1, L, S

class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int | None = None):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.kv_caches: list[TinyKvCache | None] = [None] * max_active_requests
        self.HD: tuple[int, int] | None = None
        self.last_batch_bytes = 0
        self.staging_copy_bytes = 0

    def _update_head_dim(self, H: int, D: int):
        if self.HD is None:
            self.HD = (H, D)
        elif self.HD != (H, D):
            raise ValueError(f"expect {self.HD} but got {(H, D)}")

    @staticmethod
    def _prefilled_head_dim(prefilled: TinyKvCache) -> tuple[int, int] | None:
        if not isinstance(prefilled, TinyKvFullCache):
            return None
        key_values = prefilled.key_values
        if key_values is None:
            return None
        keys, _ = key_values
        if keys.ndim != 4 or keys.shape[0] != 1:
            raise ValueError("prefilled cache must have shape [1, H, S, D]")
        return keys.shape[1], keys.shape[3]

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if keys.ndim != 4 or values.ndim != 4:
            raise ValueError("batched K/V chunks must be 4D [B, H, S, D]")
        if keys.shape != values.shape:
            raise ValueError("batched K/V chunks must have the same shape")

        B, H, S, D = keys.shape
        if B != self.max_active_requests:
            raise ValueError(
                f"expected batch size {self.max_active_requests}, got {B}"
            )
        self._update_head_dim(H, D)

        if mask_length is None:
            mask_length = S
        if mask_length < 0:
            raise ValueError("mask_length must be nonnegative")

        # Step 1: append this decode/prefill chunk into each active slot's
        # request-scoped cache, then collect the full per-request K/V.
        data: list[tuple[mx.array, mx.array, int] | None] = []
        for b in range(B):
            cache = self.kv_caches[b]
            if cache is None:
                data.append(None)
                continue
            new_keys, new_values, request_length, _ = cache.update_and_fetch(
                keys[b : b + 1],
                values[b : b + 1],
            )
            data.append((new_keys[0], new_values[0], request_length))

        # Step 2: dense batch length is the longest active request.
        batch_length = max(
            (request_length for item in data if item is not None for request_length in [item[2]]),
            default=0,
        )
        batched_keys = mx.zeros((B, H, batch_length, D), dtype=keys.dtype)
        batched_values = mx.zeros((B, H, batch_length, D), dtype=keys.dtype)

        # Start fully masked; active requests fill only their right-aligned
        # cache interval with a causal mask.
        batched_mask = mx.full(
            (B, 1, mask_length, batch_length), -mx.inf, dtype=keys.dtype
        )

        for b, item in enumerate(data):
            if item is None:
                continue
            new_keys, new_values, request_length = item
            start = batch_length - request_length
            batched_keys[b, :, start:, :] = new_keys
            batched_values[b, :, start:, :] = new_values

            if mask is None or (isinstance(mask, str) and mask == "causal"):
                batched_mask[b, :, 0:mask_length, start:] = causal_mask(
                    mask_length, request_length, dtype=keys.dtype
                )
            elif isinstance(mask, mx.array):
                batched_mask[b, :, 0:mask_length, start:] = mask
            else:
                raise ValueError(f"unsupported attention mask: {mask!r}")

        self.last_batch_bytes = batched_keys.nbytes + batched_values.nbytes
        self.staging_copy_bytes = sum(
            item[0].nbytes + item[1].nbytes for item in data if item is not None
        )
        return batched_keys, batched_values, None, batched_mask

    def update_and_fetch_paged(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        from .paged_kv_cache import PagedKvMetadata, TinyKvPagedCache

        if keys.shape != values.shape:
            raise ValueError("batched K/V chunks must have the same shape")
        B, H, S, D = keys.shape
        if B != self.max_active_requests:
            raise ValueError(
                f"expected batch size {self.max_active_requests}, got {B}"
            )

        # All active slots in one layer must share the same physical pool.
        pool = None
        for cache in self.kv_caches:
            if cache is None:
                continue
            if not isinstance(cache, TinyKvPagedCache):
                raise ValueError("paged batch requires TinyKvPagedCache slots")
            if pool is None:
                pool = cache.pool
            elif pool is not cache.pool:
                raise ValueError("paged batch slots must share one layer pool")
        if pool is None:
            raise ValueError("cannot build paged metadata without active requests")

        # Snapshot mutable state so one failing slot does not leave the other
        # slots half-appended.
        pool_state = pool._snapshot_state()
        cache_states = [
            (cache, list(cache.page_ids), list(cache.page_lens), cache.offset)
            for cache in self.kv_caches
            if cache is not None
        ]
        try:
            for b, cache in enumerate(self.kv_caches):
                if cache is not None:
                    cache.update_and_fetch_paged(keys[b : b + 1], values[b : b + 1])
        except Exception:
            pool._restore_state(pool_state)
            for cache, page_ids, page_lens, offset in cache_states:
                cache.page_ids = page_ids
                cache.page_lens = page_lens
                cache.offset = offset
            raise

        max_pages = max(
            (cache.num_pages for cache in self.kv_caches if cache is not None),
            default=0,
        )
        rows = []
        context_lens = []
        for cache in self.kv_caches:
            if cache is None:
                rows.append([-1] * max_pages)
                context_lens.append(0)
            else:
                rows.append(cache.page_ids + [-1] * (max_pages - cache.num_pages))
                context_lens.append(cache.offset)

        return PagedKvMetadata(
            key_pages=pool.key_pages,
            value_pages=pool.value_pages,
            block_table=mx.array(rows, dtype=mx.int32),
            context_lens=mx.array(context_lens, dtype=mx.int32),
            page_size=pool.page_size,
            mask=mask,
        )

    def add_request(self, prefilled: TinyKvCache, id: int):
        if not isinstance(id, int) or isinstance(id, bool) or not 0 <= id < self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        head_dim = self._prefilled_head_dim(prefilled)
        if head_dim is not None:
            self._update_head_dim(*head_dim)
        self.kv_caches[id] = prefilled

    def remove_request(self, id: int):
        if not isinstance(id, int) or isinstance(id, bool) or not 0 <= id < self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        if self.kv_caches[id] is None:
            raise ValueError(f"Request id {id} is not in the cache")
        self.kv_caches[id].release()
        self.kv_caches[id] = None


class TinyKvFullCache(TinyKvCache):
    def __init__(self, capacity: int | None = None):
        if capacity is not None and (
            not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0
        ):
            raise ValueError("capacity must be a non-negative integer or None")
        self.key_values = None
        self.offset = 0
        self.capacity = capacity
        self.logical_copy_bytes = 0
        self.physical_growth_copy_bytes = 0
        self.slice_write_bytes = 0
        self.growth_copy_bytes = 0

    @property
    def uses_capacity(self) -> bool:
        return self.capacity is not None

    def _logical_key_values(self) -> tuple[mx.array, mx.array]:
        keys, values = self.key_values
        return keys[:, :, :self.offset], values[:, :, :self.offset]

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("K/V must have matching [B, H, S, D] shapes")
        B, H, S, D = key.shape
        end = self.offset + S
        if self.uses_capacity and end > self.capacity:
            raise ValueError(
                f"KV cache capacity {self.capacity} exceeded by append ending at {end}"
            )
        if self.key_values is not None:
            keys, values = self.key_values
            storage_length = self.capacity if self.uses_capacity else self.offset
            expected = (B, H, storage_length, D)
            if (keys.shape != expected or values.shape != expected
                    or keys.dtype != key.dtype or values.dtype != value.dtype):
                raise ValueError("K/V chunk must match the existing cache layout and dtype")

        if self.uses_capacity:
            if self.key_values is None:
                self.key_values = (
                    mx.zeros((B, H, self.capacity, D), dtype=key.dtype),
                    mx.zeros((B, H, self.capacity, D), dtype=value.dtype),
                )
            if S:
                keys, values = self.key_values
                start = mx.array([self.offset], dtype=mx.int32)
                self.key_values = (
                    mx.slice_update(keys, key, start_indices=start, axes=(2,)),
                    mx.slice_update(values, value, start_indices=start, axes=(2,)),
                )
                self.slice_write_bytes += key.nbytes + value.nbytes
            self.offset = end
            keys, values = self._logical_key_values()
            return keys, values, self.offset, mask

        if self.key_values is None:
            self.key_values = (key, value)
            self.offset = key.shape[-2]
            return key, value, self.offset, mask

        prev_keys, prev_values = self.key_values
        copied_bytes = prev_keys.nbytes + prev_values.nbytes
        self.logical_copy_bytes += copied_bytes
        self.physical_growth_copy_bytes += copied_bytes
        self.growth_copy_bytes += copied_bytes

        new_keys = mx.concatenate([prev_keys, key], axis=-2)
        new_values = mx.concatenate([prev_values, value], axis=-2)
        self.key_values = (new_keys, new_values)
        self.offset += key.shape[-2]
        return new_keys, new_values, self.offset, mask

    def materialize(self):
        if self.key_values is not None:
            mx.eval(*self.key_values)

    def reset(self):
        """Reuse bounded storage; discard concatenating storage for a new request."""
        self.offset = 0
        if not self.uses_capacity:
            self.key_values = None

    def rewind(self, n: int):
        if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= self.offset:
            raise ValueError("rewind length must be between zero and the cache length")
        if n == 0:
            return
        self.offset -= n
        if self.offset == 0:
            if not self.uses_capacity:
                self.key_values = None
            return
        if self.uses_capacity:
            return
        self.key_values = (
            self.key_values[0][:, :, : self.offset],
            self.key_values[1][:, :, : self.offset],
        )
