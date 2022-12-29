import torch
import math


class CrossEntropyWithZLoss(torch.nn.Module):
    """Cross Entropy plus logit regularization via z_loss."""

    __constants__ = ["ignore_index", "z_loss_factor"]
    ignore_index: int
    z_loss_factor: float

    def __init__(self, ignore_index=-100, z_loss_factor=1e-4):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.z_loss_factor = z_loss_factor
        self.ignore_index = ignore_index

    def forward(self, inputs, labels):
        """Is this is the optimal implementation? Is this even what is meant?
        I wish there were more answers or code for PaLM

        This implementation assumes that log(Z) is log(sum(exp(logits))).
        The usage of log2 here is also a bit wild...
        """
        z_reg = inputs.exp().sum(dim=-1).log2().sum() * self.z_loss_factor
        return self.loss_fn(inputs, labels) + z_reg


class MSELoss(torch.nn.Module):
    """MSE Loss as a drop-in replacement for Cross Entropy Loss.

    This implementation includes a mean reduction in batch dimension and a 1/num_classes/M reduction in classes."""

    def __init__(self, ignore_index=-100):
        """Parameters as in Hui&Belkin, 2021, but k=1, and M=sqrt(C) (so maybe not really Hui&Belkin?)"""
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, labels):
        """Is this is the optimal implementation? Could also do an index_select variation..."""
        num_classes = inputs.shape[-1]
        valid_mask = labels != self.ignore_index
        M = math.sqrt(num_classes)
        onehot_labels = self._label_to_onehot(labels[valid_mask], M, num_classes=num_classes)
        return 1 / (2 * M * num_classes) * (inputs[valid_mask] - onehot_labels).pow(2).sum()

    @staticmethod
    @torch.jit.script
    def _label_to_onehot(target, M: float = 1.0, num_classes: int = 100):
        onehot_target = torch.zeros(target.shape[0], num_classes, device=target.device)
        onehot_target.scatter_(1, target.view(-1, 1), M)
        return onehot_target


class MSELossFast(torch.nn.Module):
    """MSE Loss as a drop-in replacement for Cross Entropy Loss. Only for 2dim inputs and 1dim labels

    This implementation includes a mean reduction in batch dimension and a 1/num_classes/M reduction in classes."""

    def __init__(self, ignore_index=-100):
        """Parameters as in Hui&Belkin, 2021, but k=1, and M=sqrt(C) (so maybe not really Hui&Belkin?)"""
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, labels):
        """Is this is the optimal implementation? This at least circumvents literal 1-hot labels"""
        num_examples, num_classes = inputs.shape
        valid_mask = labels != self.ignore_index
        M = math.sqrt(num_classes)

        inputs = inputs[valid_mask]
        labels = labels[valid_mask]

        x_i = inputs.pow(2).sum()
        x_j = inputs[torch.arange(labels.shape[-1]), labels].sum()
        return 1 / (2 * M * num_classes) * (x_i - 2 * M * x_j + labels.shape[-1] * M**2)


class L1Loss(torch.nn.Module):
    """L1 Loss as a drop-in replacement for Cross Entropy Loss. Only for 2dim inputs and 1dim labels

    This implementation includes a mean reduction in batch dimension and a 1/num_classes reduction in classes."""

    def __init__(self, ignore_index=-100):
        """."""
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, labels):
        """Optimal scaling is less clear for L1"""
        num_classes = inputs.shape[-1]
        valid_mask = labels != self.ignore_index
        M = math.sqrt(num_classes)
        onehot_labels = self._label_to_onehot(labels[valid_mask], float(num_classes), num_classes=num_classes)
        return 1 / inputs.shape[0] / M * (inputs[valid_mask] - onehot_labels).abs().sum()

    @staticmethod
    @torch.jit.script
    def _label_to_onehot(target, M: float = 1.0, num_classes: int = 100):
        onehot_target = torch.zeros(target.shape[0], num_classes, device=target.device)
        onehot_target.scatter_(1, target.view(-1, 1), M)
        return onehot_target


class SzegedyLoss(torch.nn.Module):
    """Regression directly back to input embedding. Remove the decoding layer if using this loss.

    As mentioned at https://twitter.com/ChrSzegedy/status/1533322132368728064?t=xz00T1YT3-WiE0id-h3MEA&s=19
    """

    def __init__(self, embedding_layer, ignore_index=-100, overrelaxation=2.0):
        """Overrelax parameter is quite a bit speculative..."""
        super().__init__()
        self.embedding = embedding_layer
        self.ignore_index = ignore_index
        self.overrelaxation = overrelaxation

    def forward(self, inputs, labels):
        """This really just does L2(DNN(embed(x[:,:-1]), 2.0 * stop_gradient(embed(x[:,1:]))) as quoted above"""
        num_examples, num_classes = inputs.shape
        valid_mask = labels != self.ignore_index
        M = math.sqrt(num_classes)

        inputs = inputs[valid_mask]
        with torch.no_grad():
            embedded_labels = self.overrelaxation * self.embedding(labels)[valid_mask]

        return (inputs - embedded_labels).pow(2).sum() / labels.shape[-1] / num_classes


"""Focal Loss from https://github.com/clcarwin/focal_loss_pytorch (minimally modernized into pytorch 1.12)"""

"""
MIT License

Copyright (c) 2017 carwin

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


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 5.0, size_average: bool = True, ignore_index: int = -100):
        super().__init__()
        self.register_buffer("gamma", torch.as_tensor(gamma, dtype=torch.float), persistent=False)
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_mask = target != self.ignore_index

        log_probs = torch.nn.functional.log_softmax(input[valid_mask]).gather(1, target[None, valid_mask])
        loss = -1 * (1 - log_probs.exp()) ** self.gamma * log_probs
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class IncorrectCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """CrossEntropyLoss, but only on incorrectly classified examples."""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            incorrect_preds = input.argmax(dim=-1) != target
        return torch.nn.functional.cross_entropy(
            input[incorrect_preds],
            target[incorrect_preds],
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
