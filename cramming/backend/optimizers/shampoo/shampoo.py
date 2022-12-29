"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from typing import Tuple

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer

from .shampoo_utils import (
    BlockShampooPreconditioner,
    AdagradPreconditioner,
    ShampooPreconditioner,
    LargeDimMethod,
    GraftingType,
)

logger = logging.getLogger(__name__)


class Shampoo(Optimizer):
    """Implements Shampoo algorithm.

    See details in:
    - https://arxiv.org/pdf/1802.09568.pdf
    - https://arxiv.org/pdf/2002.09018.pdf

    If root_inv_dist = True, assigns each parameter's preconditioners to different GPUs in a
    round-robin fashion.

    Uses infinity norm to evaluate residuals and errors.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (Default: 1e-2)
        betas (Tuple[float, float]): coefficients used for computing running averages
            of gradient and its square (Default: (0.9, 1.0))
        epsilon (float): term added to the denominator to improve numerical stability (Default: 1e-12)
        use_bias_correction (bool): flag for using bias correction (Default: True)
        adam_w_mode (bool): Flag for using AdamW-style weight decay (Default: True)
        weight_decay (float): weight decay (L2 penalty) (Default: 0)
        update_freq (int): frequency for updating inverse preconditioner (Default: 100)
        init_delay (int): initial delay before starting to compute root inverse (Default: 1000)
        threshold (int): threshold for switching to diagonal preconditioner (Default: 1024)
        preconditioner_dtype (torch.dtype): data type for preconditioner (Default: torch.float)
        large_dim_method (LargeDimMethod): method for handling large scale tensors. (Default: LargeDimMethod.BLOCKING)
        root_inv_dist (bool): distributes root inverse computation across multiple GPU workers (Default: True)
        use_merge_dims (bool): merge dimensions if possible while respecting threshold. (Default: True)
        grafting_type (GraftingType): Selects grafting method. (Default: GraftingType.ADAGRAD)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)
        grafting_beta2 (float): Exponential moving average factor for grafting method. (Default: 1.0)

    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 1.0),
        epsilon: float = 1e-12,
        use_bias_correction: bool = True,
        adam_w_mode: bool = True,
        weight_decay: float = 0.0,
        update_freq: int = 100,
        init_delay: int = 1000,
        threshold: int = 1024,
        preconditioner_dtype: torch.dtype = torch.float,
        large_dim_method: LargeDimMethod = LargeDimMethod.BLOCKING,
        root_inv_dist: bool = True,
        use_merge_dims: bool = True,
        grafting_type: GraftingType = GraftingType.ADAGRAD,
        grafting_epsilon: float = 1e-3,
        grafting_beta2: float = 1.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if betas[1] <= 0.0 or betas[1] > 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if grafting_beta2 <= 0.0 or grafting_beta2 > 1.0:
            raise ValueError(f"Invalid grafting beta parameter: {grafting_beta2}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if grafting_epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon value: {grafting_epsilon}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "epsilon": epsilon,
            "grafting_epsilon": grafting_epsilon,
            "grafting_beta2": grafting_beta2,
        }
        super(Shampoo, self).__init__(params, defaults)

        self.threshold = threshold
        self.update_freq = update_freq
        self.iter = 0
        self.init_delay = init_delay
        self.root_inv_dist = root_inv_dist
        self.use_merge_dims = use_merge_dims
        self.large_dim_method = large_dim_method
        self.adam_w_mode = adam_w_mode
        self.preconditioner_dtype = preconditioner_dtype
        self.use_bias_correction = use_bias_correction
        self.grafting_type = grafting_type
        self.grafting_epsilon = grafting_epsilon
        self.grafting_beta2 = grafting_beta2
        self.parameter_count = 0

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                beta1, _ = group["betas"]
                if beta1 != 0:
                    state["exp_avg"] = None

        self._initialize_preconditioners()

    @torch.no_grad()
    def _initialize_preconditioners(self):
        """Initialize Shampoo preconditioners and inverse preconditioners."""

        # iterate through each parameter (and parameter group)
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                # extract state
                state = self.state[p]
                dims = torch.tensor(p.shape)

                # Uses Adagrad if larger than threshold
                if self.large_dim_method == LargeDimMethod.ADAGRAD:
                    if torch.any(dims > self.threshold):
                        state["preconditioners"] = AdagradPreconditioner(
                            p,
                            beta2=group["betas"][1],
                            epsilon=group["epsilon"],
                            use_bias_correction=self.use_bias_correction,
                            idx=idx,
                        )
                    else:
                        state["preconditioners"] = ShampooPreconditioner(
                            p,
                            beta2=group["betas"][1],
                            epsilon=group["epsilon"],
                            use_bias_correction=self.use_bias_correction,
                            diagonal_threshold=self.threshold,
                            dtype=self.preconditioner_dtype,
                            idx=idx,
                            init_delay=self.init_delay,
                            grafting_type=self.grafting_type,
                            grafting_beta2=self.grafting_beta2,
                            grafting_epsilon=self.grafting_epsilon,
                        )

                # Uses diagonal preconditioners if larger than threshold
                elif self.large_dim_method == LargeDimMethod.DIAGONAL:
                    state["preconditioners"] = ShampooPreconditioner(
                        p,
                        beta2=group["betas"][1],
                        epsilon=group["epsilon"],
                        use_bias_correction=self.use_bias_correction,
                        diagonal_threshold=self.threshold,
                        dtype=self.preconditioner_dtype,
                        idx=idx,
                        init_delay=self.init_delay,
                        grafting_type=self.grafting_type,
                        grafting_beta2=self.grafting_beta2,
                        grafting_epsilon=self.grafting_epsilon,
                    )

                # Uses blocking if larger than threshold
                elif self.large_dim_method == LargeDimMethod.BLOCKING:
                    state["preconditioners"] = BlockShampooPreconditioner(
                        p,
                        beta2=group["betas"][1],
                        epsilon=group["epsilon"],
                        use_bias_correction=self.use_bias_correction,
                        block_size=self.threshold,
                        dtype=self.preconditioner_dtype,
                        idx=idx,
                        use_merge_dims=self.use_merge_dims,
                        init_delay=self.init_delay,
                        grafting_type=self.grafting_type,
                        grafting_beta2=self.grafting_beta2,
                        grafting_epsilon=self.grafting_epsilon,
                    )

                else:
                    raise ValueError("Large dim method " + self.large_dim_method + " is not implemented!")

                # increase parameter count
                self.parameter_count += state["preconditioners"].parameter_count

        # log total number of parameters for optimizer
        logger.info(f"Total Parameter Count: {self.parameter_count}")

    @torch.no_grad()
    def _compute_root_inverse(self):
        """Preprocesses and computes root inverse of each preconditioner. Syncs root inverse across different
        workers."""

        # loop through parameters
        for group in self.param_groups:

            # get world size
            if self.root_inv_dist:
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size = 0
                rank = None

            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                # distribute computation between workers if world_size > 0
                # NOTE: This can be further optimized by better load-balancing in the future.
                if world_size == 0 or (world_size > 0 and idx % world_size == rank):

                    # Initialize state
                    state = self.state[p]

                    # compute Shampoo preconditioner
                    if isinstance(state["preconditioners"], (ShampooPreconditioner, BlockShampooPreconditioner)):
                        state["preconditioners"].compute_root_inverse()

    @torch.no_grad()
    def _broadcast_inv_preconditioners(self):
        """Broadcasts inverse preconditioners."""

        for group in self.param_groups:
            # get world size
            world_size = dist.get_world_size()

            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                # get state, dimensions, and order
                state = self.state[p]
                src_rank = idx % world_size
                state["preconditioners"].broadcast(src_rank)

    @torch.no_grad()
    def _update_preconditioners(self):
        """Updates preconditioners.

        Note: If using L2-regularization/weight decay, it is computed within this function and therefore should not be
        recomputed elsewhere.

        """

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                weight_decay = group["weight_decay"]

                # TODO: Sparse case still not supported.
                if p.grad.is_sparse:
                    raise Exception("Sparse parameters are not currently supported by Shampoo.")

                # Dense case
                else:
                    # incorporate L2 regularization / weight decay
                    if not self.adam_w_mode and weight_decay != 0:
                        grad.add_(p, alpha=weight_decay)

                    state["preconditioners"].update_preconditioners(grad)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.iter += 1

        # update preconditioners
        self._update_preconditioners()

        # compute root inverse if delay is 0
        if self.iter % self.update_freq == 0 and self.iter >= self.init_delay:
            self._compute_root_inverse()
            if self.root_inv_dist:
                self._broadcast_inv_preconditioners()

        # perform update
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Initialize gradient, states, and dim for parameter
                grad = p.grad
                state = self.state[p]
                beta1, _ = group["betas"]
                weight_decay = group["weight_decay"]
                lr = group["lr"]

                # TODO: Sparse case still not supported.
                if p.grad.is_sparse:
                    raise Exception("Sparse parameters are not currently supported by Shampoo.")

                # Dense case
                else:

                    # incorporate momentum
                    if beta1 != 0:
                        # compute bias corrections if necessary
                        bias_correction1 = 1.0
                        if self.use_bias_correction and beta1 < 1:
                            bias_correction1 -= beta1**self.iter

                        # modify grad with momentum term
                        if state["exp_avg"] is None:
                            state["exp_avg"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                        buf = state["exp_avg"]
                        buf.mul_(beta1).add_(grad, alpha=1 - beta1)
                        grad.copy_(buf / bias_correction1)

                    # perform AdamW weight decay
                    if self.adam_w_mode and weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)

                    # compute preconditioned gradient and update parameters
                    state["preconditioners"].precondition_and_update(p, grad, lr)

        return loss
