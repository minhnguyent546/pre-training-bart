from typing import Literal

import torch
import torch.distributed as dist


class AverageMeter:
    """A class for working with average meters."""
    def __init__(
        self,
        name: str,
        value: int | float = 0.0,
        count: int = 0,
        sum: int | float = 0.0,
        device: torch.device | Literal['auto'] = 'auto',
    ) -> None:
        if count == 0:
            value = 0
            sum = 0
        self.name = name
        self.value = value
        self.count = count
        self.sum = sum
        if device == 'auto':
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

    def update(self, value: int | float, nums: int = 1) -> None:
        self.value = value
        self.sum += value * nums
        self.count += nums

    def reduce(self, dst: int) -> None:
        meters_to_reduce = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        # only `Tensor` of process with rank `dst` will be modified in-place,
        # `Tensor` of other processes will remain the same
        dist.reduce(meters_to_reduce, dst=dst, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    def all_reduce(self) -> None:
        meters_to_reduce = torch.tensor([self.sum, self.count], dtype=torch.float32, device=self.device)
        dist.all_reduce(meters_to_reduce, op=dist.ReduceOp.SUM)
        self.sum, self.count = meters_to_reduce.tolist()

    @property
    def average(self) -> float:
        try:
            return self.sum / self.count
        except ZeroDivisionError:
            return 0.0

    def reset(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def __repr__(self) -> str:
        return (
            f'{self.name}(value={self.value}, '
            f'average={self.average}, '
            f'sum={self.sum}, '
            f'count={self.count}, '
            f'device={self.device})'
        )
