"""Implementation of a progressive batching meta optimizer.
The optimizer may defer an optimization until gradient variance is small enough
"""

import torch

from collections import defaultdict
from .optimizer_modifiers import MetaOptimizer


import logging

log = logging.getLogger(__name__)
DEBUG = False


class ProgressiveBatching(MetaOptimizer):
    def __init__(self, optimizer, progress_rule="norm-based", theta=0.9, monotone=False, min_sample_guard=2, max_sample_guard=128):
        super().__init__(optimizer)

        self.progress_rule = progress_rule
        self.theta = theta
        self.monotone = monotone

        self.min_sample_guard = min_sample_guard
        self.max_sample_guard = max_sample_guard

        self.progress_state = defaultdict(dict)
        self.accumulated_steps = 0
        self.reset_sample_statistics()

    @torch.no_grad()
    def step(self):
        """(Maybe) performs a single optimization step."""
        self.update_sample_statistics()
        if self.accumulated_steps < self.min_sample_guard:
            rule_check = False
        else:
            if self.accumulated_steps > self.max_sample_guard:
                rule_check = True
            else:
                if self.progress_rule == "norm-based":
                    rule_check = self.norm_test()
                elif self.progress_rule == "inner-product":
                    rule_check = self.inner_product_test()
                elif self.progress_rule == "cov":
                    rule_check = self.coefficient_of_variation()
                elif self.progress_rule == "cosine":
                    rule_check = self.cosine_test()
                else:
                    raise ValueError(f"Invalid progress rules {self.progress_rule} given.")

        if rule_check:
            self.copy_mean_grad()  # reference running mean in p.grad attributes
            if self.monotone:
                self.min_sample_guard = self.accumulated_steps  # raise lower limit if forcing monotone batch sizes
            self.reset_sample_statistics()  # reset running mean
            super().step()
        else:
            # otherwise defer the step and accumulate more gradients
            pass

    def inner_product_test(self):
        """Inner product similar to description in Bollapragada,Byrd,Nocedal, "Adaptive Sampling Strategies for Stochastic Optimization".

        This is only a zero-memory inner product test.
        """

        global_inner_product, global_variance = 0, 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.progress_state[p]
                ndivn1 = self.accumulated_steps / (self.accumulated_steps - 1)
                corrected_mean = (state["running_mean"] - p.grad / self.accumulated_steps) * ndivn1
                global_inner_product += (p.grad * corrected_mean).sum()
                global_variance += corrected_mean.pow(2).sum()
        final_v = (global_inner_product - global_variance).pow(2)

        if DEBUG:
            inequality_repr = f"{final_v / (self.accumulated_steps - 1):10.2f} < {self.theta * global_variance**2:10.2f}"
            log.info(f"{self.accumulated_steps} - {inequality_repr}")

        return final_v / (self.accumulated_steps - 1) < self.theta * global_variance**2

    def norm_test(self):
        """Sohams version."""

        sample_var, mean_norm = 0, 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.progress_state[p]
                sample_var += state["running_variance"].sum() / (self.accumulated_steps - 1)  # bessel-corrected variance
                mean_norm += state["running_mean"].pow(2).sum()

        if DEBUG:
            log.info(f"{self.accumulated_steps} -  {sample_var / self.accumulated_steps:10.2f} < {self.theta * mean_norm:10.2f}")

        return sample_var / self.accumulated_steps < self.theta * mean_norm  # divide by |B| as in bigbatch, original version is theta=1

    def cosine_test(self):
        """Experimental."""

        total_angles, num_params = 0, 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.progress_state[p]
                ndivn1 = self.accumulated_steps / (self.accumulated_steps - 1)
                corrected_mean = (state["running_mean"] - p.grad / self.accumulated_steps) * ndivn1
                total_angles += (p.grad * corrected_mean).sum() / corrected_mean.norm() / p.grad.norm()
                num_params += 1

        average_angle = total_angles / num_params  # rather the average cosine, this not (yet) the angle

        if DEBUG:
            log.info(f"{self.accumulated_steps} -  {average_angle:10.2f} > {self.theta:10.2f}")

        return average_angle > self.theta

    def coefficient_of_variation(self):
        """unbiased cov test."""
        cov, mean_norm, num_params = 0, 0, 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.progress_state[p]
                cov += (state["running_variance"].sum() / (self.accumulated_steps - 1)).sqrt() / (state["running_mean"].pow(2).sum() + 1e-6)
                mean_norm += state["running_mean"].pow(2).sum()
                num_params += 1

        unbiased_avg_cov = (1 + 1 / (4 * self.accumulated_steps)) * cov / num_params / self.accumulated_steps

        if DEBUG:
            log.info(f"{self.accumulated_steps} -  {unbiased_avg_cov:10.2f} < {self.theta * 100:10.2f}")

        return unbiased_avg_cov < self.theta * 100

    def update_sample_statistics(self):
        """Update sample statistics based on welford accumulation. At any step variance can be finalized via running_variance / count"""
        self.accumulated_steps += 1
        for group in self.param_groups:
            for p in group["params"]:
                state = self.progress_state[p]
                current_delta = p.grad - state["running_mean"]
                state["running_mean"] += current_delta / self.accumulated_steps
                corrected_delta = p.grad - state["running_mean"]
                state["running_variance"] += current_delta * corrected_delta

    def reset_sample_statistics(self):
        """Allocate new tensors, old references are still required for the optimizer step."""
        self.last_full_step_accumulation = self.accumulated_steps + 1
        self.accumulated_steps = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.progress_state[p]
                state["running_mean"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["running_variance"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def copy_mean_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = self.progress_state[p]["running_mean"]
