"""Script for a pretraining run."""

import torch
import hydra

import os
import time
import datetime
import logging
from collections import defaultdict

import cramming

log = logging.getLogger(__name__)


def main_training_process(cfg, setup):
    """This function controls the central training loop."""
    local_time = time.time()
    model = cramming.construct_model(cfg.arch, cfg.data.vocab_size)
    dataset, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl)

    model_engine, _, _, dataloader = cramming.load_backend(
        model,
        dataset,
        tokenizer,
        cfg.train,
        cfg.impl,
        setup=setup,
    )
    model_engine.train(cfg.train.pretrain_in_train_mode)
    stats = defaultdict(list)

    # Start the clocks now:
    wallclock_timer = time.time()
    train_time = time.time()  # Crude time measurement for print_loss_every_nth_step
    training_allowed = True
    loss_vals = []

    iterable_data = enumerate(dataloader)
    if cfg.train.gradinit.enabled:
        model_engine.gradinit(iterable_data, cfg.train.optim, cfg.train.gradinit)

    # Launch training
    for step, batch in iterable_data:
        # Heavy lifting is moved to engines
        device_batch = model_engine.to_device(batch)
        loss = model_engine.step(device_batch)
        loss_vals.append(loss.detach())

        # Check stopping criteria
        if check_deadline(wallclock_timer, cfg.budget) or step == cfg.train.steps:
            training_allowed = False
            log.info("Reached deadline. Stopping training ...")

        # Collect stats and print to console and upload to wandb
        if step % cfg.impl.print_loss_every_nth_step == 0:
            loss_vals, train_time = collect_stats(step, loss_vals, train_time, stats, model_engine, dataloader, cfg)
            if check_early_termination(wallclock_timer, stats["loss"][-1], cfg.impl.early_termination):
                training_allowed = False
                log.info("Loss higher than allowed threshold. Stopping training early...")

        # Checkpointing is triggered from stopping criteria and normal intervals
        if cfg.impl.save_intermediate_checkpoints and step % cfg.impl.save_every_nth_step == 0:
            state = dict(step=step, tokenizer_name=tokenizer.name_or_path)
            checkpoint_id = loss.item()
            if cramming.utils.is_main_process():
                model_engine.save_training_checkpoint(checkpoint_id, state=state)

        if not loss.detach().isfinite():
            training_allowed = False
            log.info("Ending training due to non-finite loss.")

        flag_communication(training_allowed)

        if (cfg.dryrun and step > 2) or not training_allowed:
            break

    if cramming.utils.is_main_process():
        # Save to summary:
        metrics = dict(num_params=sum([p.numel() for p in model.parameters()]))
        cramming.utils.save_summary("pretrain", cfg, metrics, stats, time.time() - local_time, setup)
        # Save final checkpoint:
        now = datetime.datetime.now()
        checkpoint_id = f"{''.join(cfg.arch.architectures)}_{now.strftime('%Y-%m-%d')}_{loss:2.4f}"
        model_engine.save_final_model(os.path.join(cfg.base_dir, cfg.name), checkpoint_id, tokenizer, cfg.arch, cfg.dryrun)


def check_deadline(launch_time, hour_limit):
    """These measurements are deliberately wall-clock based."""
    current_time = time.time()
    return True if (current_time - launch_time) / 3600 > hour_limit else False


def check_early_termination(launch_time, loss, early_termination):
    """Early termination based on terrible loss."""
    if early_termination.enabled and loss > early_termination.loss_threshold:
        current_time = time.time()
        return True if (current_time - launch_time) / 3600 > early_termination.budget else False
    else:
        return False


def collect_stats(step, loss_vals, train_time, stats, model_engine, dataloader, cfg):
    stats["step"] += [step]
    stats["epoch"] += [dataloader.epoch_counter]

    tokens_per_step = cramming.utils.num_processes() * model_engine.record_tokens_per_step()
    stats["tokens"] += [step * tokens_per_step]
    stats["loss"] += [torch.stack(loss_vals).mean().item()]  # Averaged loss

    current_lr = model_engine.optimizer.param_groups[0]["lr"]
    log_msg = f"Train loss {loss_vals[-1].item():2.4f} at step {step} with lr {current_lr:.5f}. "
    log_msg += f"[Avg: {stats['loss'][-1]:2.4f}] "
    if step > 0:
        stats["train_time"] += [(time.time() - train_time) / cfg.impl.print_loss_every_nth_step]
        estimated_train_finish = str(datetime.timedelta(seconds=stats["train_time"][-1] * cfg.train.steps))
        tokens_per_second = tokens_per_step / stats["train_time"][-1]
        stats["tok/sec"] += [int(tokens_per_second)]
        log_msg += f" Perf: {stats['train_time'][-1]:2.4f}s per step ({tokens_per_second:.0f}t/s). "
        log_msg += f"Estimated Total Train: {estimated_train_finish}."

    # Adaptive optim stats
    stats["lr"] += [current_lr]
    stats["batch_size"] += [model_engine.record_batch_size()]
    stats["seq_length"] = [model_engine.current_seq_length]

    # Publish
    cramming.utils.wandb_log(stats, cfg)
    log.info(log_msg)

    # Clear:
    loss_vals = []
    train_time = time.time()
    return loss_vals, train_time


def flag_communication(training_allowed):
    """A quick and dirty communication through NCCL. Should not be a major burden."""
    if torch.distributed.is_initialized():
        comm_tensor = torch.as_tensor(training_allowed).cuda()
        torch.distributed.all_reduce(comm_tensor, torch.distributed.ReduceOp.MIN, async_op=False)
        if comm_tensor >= 1:
            return True
        else:
            return False
    else:
        return training_allowed


@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_training_process, job_name="pretraining")


if __name__ == "__main__":
    launch()
