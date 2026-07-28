from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from frontier.kv_cache.kv_cache_block import KVCacheBlock


@dataclass
class TieredPrefixPlan:
    query_blocks: int
    cpu_query_blocks: int
    hit_frontier_blocks: int
    gpu_blocks_by_index: dict[int, KVCacheBlock]
    cpu_block_indices: list[int]
    block_keys: list[Hashable]
    block_size: int
    prompt_tokens: int

    @property
    def gpu_hit_blocks(self) -> int:
        return len(self.gpu_blocks_by_index)

    @property
    def cpu_hit_blocks(self) -> int:
        return len(self.cpu_block_indices)

    @property
    def total_hit_blocks(self) -> int:
        return self.hit_frontier_blocks

    @property
    def hit_tokens(self) -> int:
        return self.hit_frontier_blocks * self.block_size

    @property
    def num_new_tokens(self) -> int:
        return self.prompt_tokens - self.hit_tokens

    @property
    def needs_restore(self) -> bool:
        return bool(self.cpu_block_indices)
