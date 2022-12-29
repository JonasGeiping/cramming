"""This is the original Adahessian implementation from https://github.com/amirgholami/adahessian/blob/master/image_classification/optim_adahessian.py

This snippet is from commit 935a0476aeb8f76b397d9ef4f04d59d7783abfec
"""

"""
MIT License

Copyright (c) 2020 Amir Gholaminejad, Zhewei Yao

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

import math
import torch
from torch.optim.optimizer import Optimizer


class Adahessian(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1). You can also try 0.5. For some tasks we found this to result in better performance.
        single_gpu (Bool, optional): Do you use distributed training or not "torch.nn.parallel.DistributedDataParallel" (default: True)
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4, weight_decay=0, hessian_power=1, single_gpu=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
        self.single_gpu = single_gpu
        super(Adahessian, self).__init__(params, defaults)

    def get_trace(self, params, grads):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError(
                    "Gradient tensor {:} does not have grad_fn. When calling\n".format(i)
                    + "\t\t\t  loss.backward(), make sure the option create_graph is\n"
                    + "\t\t\t  set to True."
                )

        v = [2 * torch.randint_like(p, high=2) - 1 for p in params]

        # this is for distributed setting with single node and multi-gpus,
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for v1 in v:
                dist.all_reduce(v1)
        if not self.single_gpu:
            for v_i in v:
                v_i[v_i < 0.0] = -1.0
                v_i[v_i >= 0.0] = 1.0

        hvs = torch.autograd.grad(grads, params, grad_outputs=v, only_inputs=True, retain_graph=True)

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                # Hessian diagonal block size is 1 here.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = hv.abs()

            elif len(param_size) == 4:  # Conv kernel
                # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
            hutchinson_trace.append(tmp_output)

        # this is for distributed setting with single node and multi-gpus,
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for output1 in hutchinson_trace:
                dist.all_reduce(output1 / torch.cuda.device_count())

        return hutchinson_trace

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        groups = []
        grads = []

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        # get the Hessian diagonal

        hut_traces = self.get_trace(params, grads)

        for (p, group, grad, hut_trace) in zip(params, groups, grads, hut_traces):

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = 0
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p.data)
                # Exponential moving average of Hessian diagonal square values
                state["exp_hessian_diag_sq"] = torch.zeros_like(p.data)

            exp_avg, exp_hessian_diag_sq = state["exp_avg"], state["exp_hessian_diag_sq"]

            beta1, beta2 = group["betas"]

            state["step"] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad.detach_(), alpha=1 - beta1)
            exp_hessian_diag_sq.mul_(beta2).addcmul_(hut_trace, hut_trace, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]

            # make the square root, and the Hessian power
            k = group["hessian_power"]
            denom = ((exp_hessian_diag_sq.sqrt() ** k) / math.sqrt(bias_correction2) ** k).add_(group["eps"])

            # make update
            p.data = p.data - group["lr"] * (exp_avg / bias_correction1 / denom + group["weight_decay"] * p.data)

        return loss
