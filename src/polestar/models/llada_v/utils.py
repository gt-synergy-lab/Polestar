"""Distributed-aware status output for the vision model."""

import torch.distributed as dist


def rank0_print(*args):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args)


def rank_print(*args):
    print(*args)
