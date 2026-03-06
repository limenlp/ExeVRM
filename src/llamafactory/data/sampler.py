# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterator, Optional

import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler


class PairedShuffleSampler(Sampler[int]):
    r"""Sampler that shuffles paired groups while maintaining order within each group.
    
    For paired_interleave datasets with group_size=2, the dataset is organized as:
    [A[0], B[0], A[1], B[1], A[2], B[2], ...]
    
    This sampler shuffles the pairs but keeps A before B within each pair:
    e.g., [A[2], B[2], A[0], B[0], A[1], B[1], ...]
    
    Args:
        data_source: The dataset to sample from.
        group_size: Number of samples in each group (default: 2 for paired datasets).
        generator: Optional random number generator for reproducibility.
    """

    def __init__(
        self,
        data_source,
        group_size: int = 2,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.data_source = data_source
        self.group_size = group_size
        self.generator = generator
        
        self._num_samples = len(self.data_source)
        if self._num_samples % group_size != 0:
            raise ValueError(
                f"Dataset size ({self._num_samples}) must be divisible by group_size ({group_size})."
            )
        self._num_groups = self._num_samples // group_size

    def __iter__(self) -> Iterator[int]:
        # Shuffle group indices
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        
        # Generate shuffled group order
        group_indices = torch.randperm(self._num_groups, generator=generator).tolist()
        
        # Yield sample indices in shuffled group order, maintaining order within groups
        for group_idx in group_indices:
            start_idx = group_idx * self.group_size
            for offset in range(self.group_size):
                yield start_idx + offset

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""Set the epoch for reproducible shuffling across epochs.
        
        This method is called by the Trainer at the beginning of each epoch.
        """
        if self.generator is not None:
            self.generator.manual_seed(epoch)
        else:
            self.generator = torch.Generator()
            self.generator.manual_seed(epoch)


class DistributedPairedShuffleSampler(DistributedSampler):
    r"""Distributed version of PairedShuffleSampler for multi-GPU training.

    Inherits from torch.utils.data.distributed.DistributedSampler to ensure
    compatibility with HuggingFace Trainer + Accelerate, which checks for
    DistributedSampler instances to avoid adding additional sharding.

    This sampler ensures that:
    1. Each GPU processes complete paired groups (A[i], B[i] stay together on same GPU)
    2. Groups are shuffled across the dataset
    3. Groups (not individual samples) are distributed across GPUs

    For example, with 2 datasets and 8 GPUs:
    - Total groups: N (each group has 2 samples: one from each dataset)
    - Each GPU gets N/8 groups = N/8 * 2 = N/4 samples
    - Within each GPU, samples come in pairs: A[i], B[i], A[j], B[j], ...

    Args:
        data_source: The dataset to sample from.
        group_size: Number of samples in each group (default: 2).
        num_replicas: Number of processes in distributed training.
        rank: Rank of the current process.
        shuffle: Whether to shuffle the groups.
        seed: Random seed for shuffling.
        drop_last: Whether to drop the last incomplete batch.
    """

    def __init__(
        self,
        data_source,
        group_size: int = 2,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        # Initialize parent DistributedSampler (this sets self.dataset, num_replicas, rank, etc.)
        # We pass drop_last=False to parent since we handle it ourselves for groups
        super().__init__(
            dataset=data_source,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,  # We handle drop_last for groups, not samples
        )

        self.data_source = data_source
        self.group_size = group_size
        self._drop_last = drop_last  # Store our own drop_last for groups

        self._num_samples = len(self.data_source)
        if self._num_samples % group_size != 0:
            raise ValueError(
                f"Dataset size ({self._num_samples}) must be divisible by group_size ({group_size})."
            )
        self._num_groups = self._num_samples // group_size

        # Distribute GROUPS across replicas (not samples)
        # This ensures each GPU gets complete pairs
        if self._drop_last and self._num_groups % self.num_replicas != 0:
            self._num_groups_per_replica = self._num_groups // self.num_replicas
        else:
            self._num_groups_per_replica = (self._num_groups + self.num_replicas - 1) // self.num_replicas

        self._total_groups = self._num_groups_per_replica * self.num_replicas
        # Each replica processes this many samples
        self._num_samples_per_replica = self._num_groups_per_replica * group_size

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            group_indices = torch.randperm(self._num_groups, generator=g).tolist()
        else:
            group_indices = list(range(self._num_groups))

        # Pad groups to make evenly divisible across replicas
        if len(group_indices) < self._total_groups:
            padding_size = self._total_groups - len(group_indices)
            group_indices += group_indices[:padding_size]

        # Distribute groups across replicas (stride by num_replicas)
        # GPU 0 gets groups [0, 8, 16, ...], GPU 1 gets [1, 9, 17, ...], etc.
        my_group_indices = group_indices[self.rank:self._total_groups:self.num_replicas]

        # Expand groups to sample indices (maintaining A, B order within each group)
        indices = []
        for group_idx in my_group_indices:
            start_idx = group_idx * self.group_size
            for offset in range(self.group_size):
                indices.append(start_idx + offset)

        return iter(indices)

    def __len__(self) -> int:
        return self._num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        r"""Set the epoch for reproducible shuffling across epochs."""
        self.epoch = epoch

