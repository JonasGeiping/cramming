"""Pre-Norm / Post-norm / sandwich fused layers with dropout."""

import torch
from torch.nn.functional import dropout
from functools import partial


def get_layer_fn(type="pre", prob=0.1, scripting=True, dn=False, drop=False):
    if not dn and not drop:
        base_train, base_eval = simplified_layer_training, simplified_layer_eval
    else:
        base_train, base_eval = scaled_layer_training, scaled_layer_eval
    if type in ["pre", "post"]:
        if scripting:
            fn_train, fn_eval = torch.jit.script(base_train), torch.jit.script(base_eval)
        else:
            fn_train, fn_eval = base_train, base_eval
        return partial(fn_train, prob=prob), partial(fn_eval, prob=prob)
    elif type == "sandwich":
        return torch.jit.script(sandwich_layer_structure) if scripting else sandwich_layer_structure
    else:
        raise ValueError("Invalid layer type.")


def layer_structure(states, outputs, alpha, residual_scale, prob: float = 0.1, training: bool = False):
    return states * alpha + residual_scale * dropout(outputs, p=prob, training=training)


def scaled_layer_training(states, outputs, alpha, residual_scale, prob: float = 0.1):
    return layer_structure(states, outputs, alpha, residual_scale, prob, training=True)


def scaled_layer_eval(states, outputs, alpha, residual_scale, prob: float = 0.1):
    return layer_structure(states, outputs, alpha, residual_scale, prob, training=False)


def sandwich_layer_structure(states, outputs, alpha, residual_scale, prob: float = 0.1, training: bool = False):
    states = states * alpha + residual_scale * outputs
    return states


def simplified_layer_structure(states, outputs, alpha, residual_scale, prob: float = 0.1, training: bool = False):
    return states + dropout(outputs, p=prob, training=training)


def simplified_layer_training(states, outputs, alpha, residual_scale, prob: float = 0.1):
    return simplified_layer_structure(states, outputs, alpha, residual_scale, prob, training=True)


def simplified_layer_eval(states, outputs, alpha, residual_scale, prob: float = 0.1):
    return simplified_layer_structure(states, outputs, alpha, residual_scale, prob, training=False)
