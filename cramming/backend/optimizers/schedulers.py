"""Misc. optimizer implementations."""
import torch
import transformers
import math

from torch.optim.lr_scheduler import LambdaLR
import time
from functools import partial


def get_schedule_fn(cfg_train):
    """Returns a callable scheduler_fn(optimizer).

    Todo: Sanitize and unify these schedulers...
    """
    if (cfg_train.warmup_steps) > 0 and (cfg_train.warmup_steps < 1):
        # warmup could be a percentage in which case this line converts to steps again
        cfg_train.warmup_steps = int(cfg_train.warmup_steps * cfg_train.steps)

    # Load huggingface schedulers based on total steps
    if cfg_train.scheduler == "polynomial-decay":
        scheduler_fn = partial(
            transformers.get_polynomial_decay_schedule_with_warmup,
            num_warmup_steps=cfg_train.warmup_steps,
            num_training_steps=cfg_train.steps,
            lr_end=1e-7,
            power=1.0,
        )
    elif cfg_train.scheduler == "cosine-decay":
        scheduler_fn = partial(
            transformers.get_cosine_schedule_with_warmup,
            num_warmup_steps=cfg_train.warmup_steps,
            num_training_steps=cfg_train.steps,
            num_cycles=0.5,
        )
    elif cfg_train.scheduler == "inverse-sqrt":
        scheduler_fn = partial(
            get_inverse_sqrt_scheduler,
            num_warmup_steps=cfg_train.warmup_steps,
            num_training_steps=cfg_train.steps,
        )
    elif cfg_train.scheduler == "budget-inverse-sqrt":
        scheduler_fn = partial(
            get_budget_inv_sqrt_scheduler,
            hour_budget=cfg_train.budget,
            num_warmup_steps=cfg_train.warmup_steps,
            num_training_steps=cfg_train.steps,
        )
    elif cfg_train.scheduler == "one-cycle":  # this is a simplified one-cycle
        scheduler_fn = partial(
            get_one_cycle,
            num_training_steps=cfg_train.steps,
        )
    elif cfg_train.scheduler == "ramp":  # this is a simplified one-cycle
        scheduler_fn = partial(
            get_ramp,
            num_training_steps=cfg_train.steps,
        )
    elif cfg_train.scheduler == "budget-inverse-sqrt":
        scheduler_fn = partial(
            get_budget_inv_sqrt_scheduler,
            hour_budget=cfg_train.budget,
            num_warmup_steps=cfg_train.warmup_steps,
            num_training_steps=cfg_train.steps,
        )
    elif cfg_train.scheduler == "budget-cosine-decay":
        scheduler_fn = partial(
            get_budget_cosine_schedule_with_warmup,
            hour_budget=cfg_train.budget,
            num_warmup_steps=cfg_train.warmup_steps,
            num_training_steps=cfg_train.steps,
        )
    elif cfg_train.scheduler == "budget-linear":
        scheduler_fn = partial(
            get_budget_linear_schedule_with_warmup,
            hour_budget=cfg_train.budget,
            num_warmup_steps=cfg_train.warmup_steps,
            num_training_steps=cfg_train.steps,
        )
    elif cfg_train.scheduler == "budget-one-cycle":  # this is a simplified one-cycle
        scheduler_fn = partial(
            get_budget_one_cycle,
            hour_budget=cfg_train.budget,
            num_training_steps=cfg_train.steps,
        )
    elif cfg_train.scheduler == "budget-ramp":
        scheduler_fn = partial(
            get_budget_ramp,
            hour_budget=cfg_train.budget,
            num_training_steps=cfg_train.steps,
        )
    elif cfg_train.scheduler == "budget-inv-cosine":
        scheduler_fn = partial(
            get_budget_inv_cosine_schedule,
            hour_budget=cfg_train.budget,
            num_training_steps=cfg_train.steps,
        )
    elif cfg_train.scheduler == "budget-dive":
        scheduler_fn = partial(
            get_budget_dive,
            hour_budget=cfg_train.budget,
            num_training_steps=cfg_train.steps,
            falloff=0.5,
        )
    elif cfg_train.scheduler == "budget-dive-slow":
        scheduler_fn = partial(
            get_budget_dive,
            hour_budget=cfg_train.budget,
            num_training_steps=cfg_train.steps,
            falloff=0.75,
        )
    elif cfg_train.scheduler == "budget-dive-fast":
        scheduler_fn = partial(
            get_budget_dive,
            hour_budget=cfg_train.budget,
            num_training_steps=cfg_train.steps,
            falloff=0.25,
        )
    elif cfg_train.scheduler == "triangle1":
        scheduler_fn = partial(
            get_budget_triangle,
            hour_budget=cfg_train.budget,
            num_training_steps=cfg_train.steps,
            falloff=0.25,
            base_percentage=0.5,
        )
    elif cfg_train.scheduler == "triangle2":
        scheduler_fn = partial(
            get_budget_triangle,
            hour_budget=cfg_train.budget,
            num_training_steps=cfg_train.steps,
            falloff=0.25,
            base_percentage=0.25,
        )
    elif cfg_train.scheduler in [
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ]:

        def scheduler_fn(optimizer):
            return transformers.get_scheduler(
                name=cfg_train.scheduler,
                optimizer=optimizer,
                num_warmup_steps=cfg_train.warmup_steps,
                num_training_steps=cfg_train.steps,
            )

    elif cfg_train.scheduler == "none" or cfg_train.scheduler is None:
        scheduler_fn = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[], gamma=1)
    else:
        raise ValueError(f"Invalid schedule {cfg_train.scheduler} given.")
    return scheduler_fn


"""FairSeq-like inverse-square-root scheduler:"""


def get_inverse_sqrt_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`--warmup-init-lr`) until the configured
    learning rate (`--lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]
    After warmup:
      lr = decay_factor / sqrt(update_num)
    where
      decay_factor = args.lr * sqrt(args.warmup_updates)
    """
    # linearly warmup for the first args.warmup_updates
    lr_step = 1 / num_warmup_steps
    # then, decay prop. to the inverse square root of the update number
    decay_factor = num_warmup_steps**0.5

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step * lr_step)
        else:
            return float(decay_factor * current_step**-0.5)

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def get_one_cycle(optimizer, num_training_steps):
    """Simple single-cycle scheduler. Not including paper/fastai three-phase things or asymmetry."""
    max_lr = optimizer.defaults["lr"]

    def lr_lambda(current_step):
        if current_step < num_training_steps / 2:
            return float(current_step / (num_training_steps / 2))
        else:
            return float(2 - current_step / (num_training_steps / 2))

    return LambdaLR(optimizer, lr_lambda, -1)


def get_ramp(optimizer, num_training_steps):
    """to the MOON."""
    max_lr = optimizer.defaults["lr"]

    def lr_lambda(current_step):
        return float(current_step / num_training_steps)

    return LambdaLR(optimizer, lr_lambda, -1)


"""Wallclock time schedulers."""


def get_budget_inv_sqrt_scheduler(optimizer, hour_budget, num_warmup_steps, num_training_steps):
    """Time-based scheduler as described in Iszak et al. plus inv_sqrt.
    Takes in num_warmup_steps and num_training_steps as normal, but actually squeezes the planned schedule into the
    budget given by hour_budget, based on wallclock measurements.

    Reference: https://github.com/IntelLabs/academic-budget-bert/blob/main/pretraining/schedules.py
    """
    lr_step = 1 / num_warmup_steps
    decay_factor = num_warmup_steps**0.5

    initial_time = time.time()

    def lr_lambda(current_step: int):
        elapsed_hours = (time.time() - initial_time) / 60 / 60
        fake_step = int(elapsed_hours / hour_budget * num_training_steps)

        if fake_step < num_warmup_steps:
            return float(fake_step * lr_step)
        else:
            return float(decay_factor * fake_step**-0.5)

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def get_budget_linear_schedule_with_warmup(optimizer, hour_budget, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Follows the huggingface transformers scheduler with the same name, but gets an additional arg hour_budget"""
    initial_time = time.time()

    def lr_lambda(current_step):
        elapsed_hours = (time.time() - initial_time) / 60 / 60
        if current_step == 0:
            fake_step = 0
        else:
            fake_step = int(elapsed_hours / hour_budget * num_training_steps)

        if fake_step < num_warmup_steps:
            return float(fake_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - fake_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, -1)


def get_budget_cosine_schedule_with_warmup(optimizer, hour_budget, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Follows the huggingface transformers scheduler with the same name, but gets an additional arg hour_budget"""
    initial_time = time.time()

    def lr_lambda(current_step):
        elapsed_hours = (time.time() - initial_time) / 60 / 60
        if current_step == 0:
            fake_step = 0
        else:
            fake_step = int(elapsed_hours / hour_budget * num_training_steps)
        if fake_step < num_warmup_steps:
            return float(fake_step) / float(max(1, num_warmup_steps))
        progress = float(fake_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)


def get_budget_one_cycle(optimizer, hour_budget, num_training_steps):
    """Simple single-cycle scheduler. Not including paper/fastai three-phase things or asymmetry."""
    initial_time = time.time()

    def lr_lambda(current_step):
        elapsed_hours = (time.time() - initial_time) / 60 / 60
        if current_step == 0:
            fake_step = 0
        else:
            fake_step = int(elapsed_hours / hour_budget * num_training_steps)
        if fake_step < num_training_steps / 2:
            return float(fake_step / (num_training_steps / 2))
        else:
            return float(2 - fake_step / (num_training_steps / 2))

    return LambdaLR(optimizer, lr_lambda, -1)


def get_budget_ramp(optimizer, hour_budget, num_training_steps):
    """to the moon."""
    initial_time = time.time()

    def lr_lambda(current_step):
        elapsed_hours = (time.time() - initial_time) / 60 / 60
        fake_step = int(elapsed_hours / hour_budget * num_training_steps)
        return float(fake_step / num_training_steps)

    return LambdaLR(optimizer, lr_lambda, -1)


def get_budget_inv_cosine_schedule(optimizer, hour_budget, num_training_steps, num_cycles=0.5):
    """An inverse cosine schedule, with limited budget."""
    initial_time = time.time()

    def lr_lambda(current_step):
        elapsed_hours = (time.time() - initial_time) / 60 / 60
        if current_step == 0:
            fake_step = 0
        else:
            fake_step = int(elapsed_hours / hour_budget * num_training_steps)

        progress = 1 - fake_step / float(max(1, num_training_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, -1)


def get_budget_triangle(optimizer, hour_budget, num_training_steps, base_percentage=0.5, falloff=0.5):
    """Linear increase from a percentage of the base learning rate, then linear decay.

    plot min(0.5 + x * (1 - 0.5)/(1-0.25) / 1000, 1/0.25 - x / (1000 * 0.25)) from 0 to 1000 in the plot range 0 to 1
    """
    initial_time = time.time()

    def lr_lambda(current_step):
        elapsed_hours = (time.time() - initial_time) / 60 / 60
        if current_step == 0:
            fake_step = 0
        else:
            fake_step = int(elapsed_hours / hour_budget * num_training_steps)
        return min(
            base_percentage + fake_step * (1 - base_percentage) / (1 - falloff) / num_training_steps,
            float(1 / falloff - fake_step / (num_training_steps * falloff)),
        )

    return LambdaLR(optimizer, lr_lambda, -1)


def get_budget_dive(optimizer, hour_budget, num_training_steps, falloff=0.5):
    """Constant, then linear decay.
    plot min(1, 1/0.5 - x / (1000 * 0.5)) from 0 to 1000 in the plot range 0 to 1
    """
    initial_time = time.time()

    def lr_lambda(current_step):
        elapsed_hours = (time.time() - initial_time) / 60 / 60
        if current_step == 0:
            fake_step = 0
        else:
            fake_step = int(elapsed_hours / hour_budget * num_training_steps)
        return min(1.0, float(1 / falloff - fake_step / (num_training_steps * falloff)))

    return LambdaLR(optimizer, lr_lambda, -1)
