from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from math import ceil
from typing import Optional

from frontier.entities.request import Request
from frontier.kv_cache.kv_cache_block import KVCacheBlock
from frontier.kv_cache.kv_cache_block_pool import BlockPool


@dataclass
class PrefixCacheStats:
    reset: bool = False
    admissions: int = 0
    queries: int = 0
    hits: int = 0

    @property
    def requests(self) -> int:
        """Backward-compatible alias for admission-scoped request counts."""
        return self.admissions


class KVCacheManager:
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        enable_caching: bool,
        caching_hash_algo: str,
        num_preallocate_tokens: int,
        caching_key_mode: str = "block_hash",
    ) -> None:
        self.block_size = int(block_size)
        self.num_gpu_blocks = int(num_gpu_blocks)
        self.enable_caching = bool(enable_caching)
        self.caching_hash_algo = str(caching_hash_algo)
        self.caching_key_mode = str(caching_key_mode)
        if self.caching_key_mode not in {"block_hash", "session"}:
            raise ValueError(
                "caching_key_mode must be 'block_hash' or 'session', "
                f"got={self.caching_key_mode!r}"
            )
        self.num_preallocate_tokens = int(num_preallocate_tokens)
        self.num_preallocate_blocks = (
            ceil(self.num_preallocate_tokens / self.block_size)
            if self.num_preallocate_tokens > 0
            else 0
        )
        self.block_pool = BlockPool(self.num_gpu_blocks, self.enable_caching)
        self.req_to_blocks: defaultdict[int, list[KVCacheBlock]] = defaultdict(list)
        self.num_cached_blocks: dict[int, int] = {}
        self.prefix_cache_stats = PrefixCacheStats()

    @property
    def usage(self) -> float:
        return self.block_pool.get_usage()

    @property
    def num_used_blocks(self) -> int:
        return self.block_pool.get_num_used_blocks()

    def get_num_blocks_for_request(self, request: Request) -> int:
        return len(self.req_to_blocks.get(request.id, []))

    def make_prefix_cache_stats(self) -> PrefixCacheStats:
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def _get_session_block_keys(
        self,
        request: Request,
        num_blocks: int,
        start_block_index: int = 0,
    ) -> list[Hashable]:
        if request.session_id is None:
            raise ValueError(
                "session_id is required when prefix caching key mode is 'session'"
            )
        session_id = int(request.session_id)
        start_block_index = max(int(start_block_index), 0)
        num_blocks = max(int(num_blocks), 0)
        return [
            ("session", session_id, block_index)
            for block_index in range(
                start_block_index,
                start_block_index + num_blocks,
            )
        ]

    def _get_request_lookup_block_keys(
        self,
        request: Request,
    ) -> list[Hashable]:
        if self.caching_key_mode == "block_hash":
            return list(request.block_hash_ids or [])
        num_prompt_blocks = int(request.num_prefill_tokens) // self.block_size
        return self._get_session_block_keys(request, num_prompt_blocks)

    def get_request_lookup_block_keys(
        self, request: Request
    ) -> list[Hashable]:
        """Return the full prompt lookup-key sequence for tiered planning."""
        return self._get_request_lookup_block_keys(request)

    def get_cached_block_for_key(
        self, block_key: Hashable
    ) -> Optional[KVCacheBlock]:
        if not self.enable_caching:
            return None
        return self.block_pool.get_cached_block(block_key)

    def _get_num_request_storage_block_keys(
        self,
        request: Request,
    ) -> int:
        if self.caching_key_mode == "block_hash":
            return len(request.block_hash_ids or [])
        return int(request.total_tokens) // self.block_size

    def _get_request_storage_block_keys(
        self,
        request: Request,
        *,
        start_block_index: int,
        end_block_index: int,
    ) -> list[Hashable]:
        if end_block_index <= start_block_index:
            return []
        if self.caching_key_mode == "block_hash":
            return list(
                (request.block_hash_ids or [])[
                    start_block_index:end_block_index
                ]
            )
        return self._get_session_block_keys(
            request,
            end_block_index - start_block_index,
            start_block_index=start_block_index,
        )

    def get_computed_blocks(
        self,
        request: Request,
    ) -> tuple[list[KVCacheBlock], int, int]:
        if not self.enable_caching:
            return [], 0, 0

        block_hashes = self._get_request_lookup_block_keys(request)
        computed_blocks: list[KVCacheBlock] = []
        for block_hash in block_hashes:
            cached_block = self.block_pool.get_cached_block(block_hash)
            if cached_block is None:
                break
            computed_blocks.append(cached_block)

        return (
            computed_blocks,
            len(computed_blocks) * self.block_size,
            len(block_hashes),
        )

    def record_prefix_cache_admission(
        self,
        *,
        query_blocks: int,
        hit_blocks: int,
    ) -> None:
        if query_blocks < 0:
            raise ValueError(
                f"query_blocks must be >= 0, got={query_blocks}"
            )
        if hit_blocks < 0 or hit_blocks > query_blocks:
            raise ValueError(
                "hit_blocks must be between 0 and query_blocks, "
                f"got hits={hit_blocks}, queries={query_blocks}"
            )
        self.prefix_cache_stats.admissions += 1
        self.prefix_cache_stats.queries += int(query_blocks)
        self.prefix_cache_stats.hits += int(hit_blocks)

    def _get_num_new_blocks_required(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
    ) -> int:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be > 0")

        computed_blocks = new_computed_blocks or []
        num_computed_tokens = int(request.num_processed_tokens) + len(
            computed_blocks
        ) * self.block_size
        num_required_blocks = ceil((num_computed_tokens + num_tokens) / self.block_size)
        current_blocks = self.req_to_blocks[request.id]
        return num_required_blocks - len(current_blocks) - len(computed_blocks)

    def can_allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
    ) -> bool:
        num_new_blocks = self._get_num_new_blocks_required(
            request,
            num_tokens,
            new_computed_blocks,
        )
        evictable_computed_blocks = sum(
            1 for block in (new_computed_blocks or []) if block.ref_cnt == 0
        )
        return num_new_blocks <= (
            self.block_pool.get_num_free_blocks() - evictable_computed_blocks
        )

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional[list[KVCacheBlock]] = None,
    ) -> Optional[list[KVCacheBlock]]:
        computed_blocks = list(new_computed_blocks or [])
        if not self.can_allocate_slots(request, num_tokens, computed_blocks):
            return None

        if self.enable_caching:
            self.block_pool.touch(computed_blocks)
        elif computed_blocks:
            raise ValueError(
                "Computed prefix blocks are not allowed when prefix caching is disabled."
            )

        num_new_blocks = self._get_num_new_blocks_required(
            request,
            num_tokens,
            computed_blocks,
        )
        request_blocks = self.req_to_blocks[request.id]
        request_blocks.extend(computed_blocks)
        if num_new_blocks > 0:
            num_new_blocks = min(
                num_new_blocks + self.num_preallocate_blocks,
                self.block_pool.get_num_free_blocks(),
            )
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            request_blocks.extend(new_blocks)
        else:
            new_blocks = []

        if not self.enable_caching:
            return new_blocks

        # Allocation only reserves space. Newly allocated blocks must not become
        # prefix-cache hits until the batch that computes them has completed.
        # Prefix blocks returned by get_computed_blocks() are already ready.
        self.num_cached_blocks.setdefault(request.id, len(computed_blocks))
        return new_blocks

    def can_allocate_tiered_prefix_for_admission(
        self,
        *,
        gpu_blocks_by_index: dict[int, KVCacheBlock],
        cpu_restore_count: int,
        suffix_reservation_count: int = 0,
        minimum_free_blocks_after_reservation: int = 0,
    ) -> bool:
        """Return whether a staged mixed prefix can be admitted atomically."""
        if cpu_restore_count < 0:
            raise ValueError("cpu_restore_count must be >= 0")
        if suffix_reservation_count < 0:
            raise ValueError("suffix_reservation_count must be >= 0")
        if minimum_free_blocks_after_reservation < 0:
            raise ValueError(
                "minimum_free_blocks_after_reservation must be >= 0"
            )
        evictable_gpu_hits = sum(
            1
            for block in gpu_blocks_by_index.values()
            if block.ref_cnt == 0
        )
        return int(cpu_restore_count) + int(suffix_reservation_count) <= (
            self.block_pool.get_num_free_blocks()
            - evictable_gpu_hits
            - int(minimum_free_blocks_after_reservation)
        )

    def allocate_tiered_prefix_for_admission(
        self,
        request: Request,
        *,
        hit_frontier_blocks: int,
        gpu_blocks_by_index: dict[int, KVCacheBlock],
        cpu_restore_indices: list[int],
        suffix_reservation_count: int = 0,
    ) -> dict[int, KVCacheBlock]:
        """Materialize a staged mixed prefix during scheduler admission."""
        hit_frontier_blocks = int(hit_frontier_blocks)
        cpu_restore_indices = [int(index) for index in cpu_restore_indices]
        if request.id in self.req_to_blocks and self.req_to_blocks[request.id]:
            raise ValueError(
                f"Request {request.id} already owns KV blocks before tiered restore"
            )
        expected_indices = set(range(hit_frontier_blocks))
        actual_indices = set(gpu_blocks_by_index) | set(cpu_restore_indices)
        if actual_indices != expected_indices:
            raise ValueError(
                "Tiered prefix sources must cover the full hit frontier: "
                f"expected={sorted(expected_indices)}, actual={sorted(actual_indices)}"
            )
        if set(gpu_blocks_by_index) & set(cpu_restore_indices):
            raise ValueError("GPU and CPU prefix sources overlap")
        if not self.can_allocate_tiered_prefix_for_admission(
            gpu_blocks_by_index=gpu_blocks_by_index,
            cpu_restore_count=len(cpu_restore_indices),
            suffix_reservation_count=suffix_reservation_count,
        ):
            raise ValueError(
                f"Insufficient GPU blocks to reserve CPU restore for request {request.id}"
            )

        self.block_pool.touch(list(gpu_blocks_by_index.values()))
        restore_blocks = self.block_pool.get_new_blocks(len(cpu_restore_indices))
        restore_by_index = dict(zip(cpu_restore_indices, restore_blocks))
        ordered_blocks = [
            gpu_blocks_by_index.get(index, restore_by_index.get(index))
            for index in range(hit_frontier_blocks)
        ]
        if any(block is None for block in ordered_blocks):
            raise AssertionError("Tiered prefix reservation produced a missing block")
        self.req_to_blocks[request.id].extend(ordered_blocks)
        if suffix_reservation_count:
            self.req_to_blocks[request.id].extend(
                self.block_pool.get_new_blocks(
                    int(suffix_reservation_count)
                )
            )
        self.num_cached_blocks[request.id] = len(gpu_blocks_by_index)
        return restore_by_index

    def publish_restored_prefix(
        self,
        request: Request,
        *,
        restore_blocks_by_index: dict[int, KVCacheBlock],
        block_keys: list[Hashable],
        hit_frontier_blocks: int,
    ) -> None:
        """Publish restored CPU blocks as ready GPU prefix-cache entries."""
        for block_index, block in sorted(restore_blocks_by_index.items()):
            if block_index >= len(block_keys):
                raise ValueError(
                    f"Missing block key for restored block index {block_index}"
                )
            if block.block_hash is None:
                block.block_hash = block_keys[block_index]
                self.block_pool.cached_block_hash_to_block[
                    block.block_hash
                ][block.block_id] = block
        self.num_cached_blocks[request.id] = int(hit_frontier_blocks)

    def mark_blocks_computed(self, request: Request) -> None:
        """Publish full blocks up to the request's completed-token frontier."""
        if not self.enable_caching:
            return

        request_blocks = self.req_to_blocks.get(request.id, [])
        if not request_blocks:
            return

        num_ready_blocks = min(
            int(request.num_processed_tokens) // self.block_size,
            len(request_blocks),
            self._get_num_request_storage_block_keys(request),
        )
        num_cached_blocks = self.num_cached_blocks.get(request.id, 0)
        if num_ready_blocks <= num_cached_blocks:
            return

        new_block_keys = self._get_request_storage_block_keys(
            request,
            start_block_index=num_cached_blocks,
            end_block_index=num_ready_blocks,
        )
        self.block_pool.cache_full_blocks(
            blocks=request_blocks[num_cached_blocks:num_ready_blocks],
            block_hashes=new_block_keys,
            num_cached_blocks=0,
            num_full_blocks=len(new_block_keys),
        )
        self.num_cached_blocks[request.id] = num_ready_blocks

    def free(self, request: Request) -> None:
        request_blocks = self.req_to_blocks.pop(request.id, [])
        if request_blocks:
            self.block_pool.free_blocks(reversed(request_blocks))
        self.num_cached_blocks.pop(request.id, None)

    def free_block_hashes(self, request: Request) -> None:
        self.num_cached_blocks.pop(request.id, None)
