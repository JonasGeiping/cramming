"""Code from https://github.com/jxbz/agd, rewritten by me to fit closer to the standard pytorch format."""

import math
import torch

from torch.optim.optimizer import Optimizer

import functools
from typing import Iterable, Tuple


@functools.cache
def singular_value(p_shape: Tuple[int]):
    """requires hashable input"""
    if len(p_shape) == 1:
        return 1.0
    sv = math.sqrt(p_shape[0] / p_shape[1])
    if len(p_shape) == 4:
        sv /= math.sqrt(p_shape[2] * p_shape[3])
    return sv


class AGD(Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter], gain: float = 1.0, depth: int = 16, **kwargs):
        """Set depth to len(list(model.parameters()))."""

        defaults = dict(gain=gain, depth=depth)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        G = 0
        for group in self.param_groups:
            depth, gain = group["depth"], group["gain"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Please no")
                if p.dim() > 1:
                    G += singular_value(p.shape) * grad.norm(dim=(0, 1)).sum()

        G /= depth
        log = math.log(0.5 * (1 + math.sqrt(1 + 4 * G)))

        for group in self.param_groups:
            depth, gain = group["depth"], group["gain"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if p.dim() > 1:
                    factor = singular_value(p.shape) / grad.norm(dim=(0, 1), keepdim=True)
                else:
                    factor = 1.0  # crime
                p -= gain * log / depth * factor * grad

        return loss
