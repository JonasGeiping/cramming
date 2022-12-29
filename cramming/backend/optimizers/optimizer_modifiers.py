"""This is the apex LARS implementation, from the apex repository.

It implements LARS + optional clipping

https://github.com/NVIDIA/apex/blob/d74fda260c403f775817470d87f810f816f3d615/apex/parallel/LARC.py


I did rename it to "LARS".
"""

import torch


class MetaOptimizer(torch.optim.Optimizer):
    """base class for a meta optimizer that wraps and modifies an existing pytorch optimizer."""

    def __init__(self, optimizer):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    def __repr__(self):
        return self.__class__.__name__ + self.optim.__repr__()

    def __getattr__(self, name):
        """Call this only if all other attributes are exhausted."""
        return getattr(self.optim, name)

    @torch.no_grad()
    def step(self, closure=None):
        return self.optim.step(closure)


class LARS(MetaOptimizer):
    """
    :class:`LARS` [LARC in apex] is a pytorch implementation of both the scaling and clipping variants of LARS,
    in which the ratio between gradient and parameter magnitudes is used to calculate an adaptive
    local learning rate for each individual parameter. The algorithm is designed to improve
    convergence of large batch training.

    See https://arxiv.org/abs/1708.03888 for calculation of the local learning rate.

    In practice it modifies the gradients of parameters as a proxy for modifying the learning rate
    of the parameters. This design allows it to be used as a wrapper around any torch.optim Optimizer.

    ```
    model = ...
    optim = torch.optim.Adam(model.parameters(), lr=...)
    optim = LARS(optim)
    ```

    Args:
        optimizer: Pytorch optimizer to wrap and modify learning rate for.
        trust_coefficient: Trust coefficient for calculating the lr. See https://arxiv.org/abs/1708.03888
        clip: Decides between clipping or scaling mode of LARC [LARS + clip].
              If `clip=True` the learning rate is set to `min(optimizer_lr, local_lr)` for each parameter.
              If `clip=False` the learning rate is set to `local_lr*optimizer_lr`.
        eps: epsilon kludge to help with numerical stability while calculating adaptive_lr
    """

    def __init__(self, optimizer, trust_coefficient=0.02, clip=False, eps=1e-8):
        self.param_groups = optimizer.param_groups
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip

    def step(self, closure=None):
        loss = None
        with torch.no_grad():
            weight_decays = []
            for group in self.optim.param_groups:
                # absorb weight decay control from optimizer
                weight_decay = group["weight_decay"] if "weight_decay" in group else 0
                weight_decays.append(weight_decay)
                group["weight_decay"] = 0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        # calculate adaptive lr + weight decay
                        adaptive_lr = self.trust_coefficient * (param_norm) / (grad_norm + param_norm * weight_decay + self.eps)

                        # clip learning rate for LARC
                        if self.clip:
                            # calculation of adaptive_lr so that when multiplied by lr it equals `min(adaptive_lr, lr)`
                            adaptive_lr = min(adaptive_lr / group["lr"], 1)

                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        loss = self.optim.step(closure)
        # return weight decay control to optimizer
        for i, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[i]

        return loss


"""This the SAM pytorch implementation from https://github.com/davda54/sam
with a minor modification """

"""
MIT License
Copyright (c) 2021 David Samuel
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class SAM(MetaOptimizer):
    def __init__(self, base_optimizer_instance, rho=0.05):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = rho

        self.optim = base_optimizer_instance
        self.param_groups = base_optimizer_instance.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.optim.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        closure()
        self.first_step(zero_grad=True)
        loss = closure()
        self.second_step()
        return loss

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group["params"] if p.grad is not None]),
            p=2,
        )
        return norm
