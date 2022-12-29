"""Script to evaluate a pretrained model."""

import torch
import hydra


import time
import datetime
import logging
from collections import defaultdict

import cramming
import evaluate


log = logging.getLogger(__name__)


def main_downstream_process(cfg, setup):
    """This function controls the central routine."""
    local_time = time.time()

    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)
    tasks = cramming.prepare_task_dataloaders(tokenizer, cfg.eval, cfg.impl)

    metrics = dict()
    stats = defaultdict(list)
    # Start the clocks now:
    for task_name, task in tasks.items():
        cfg.eval.steps = len(task["trainloader"]) * cfg.eval.epochs
        log.info(f"Finetuning task {task_name} with {task['num_classes']} classes for {cfg.eval.steps} steps.")
        # Prepare model for finetuning:
        model = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=task["num_classes"])
        model_engine, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.eval, cfg.impl, setup=setup)
        model_engine.load_checkpoint(cfg_arch, model_file)
        metric = evaluate.load(task["details"]["collection"], task_name, cache_dir=cfg.impl.path)
        # Launch training
        model_engine.train()
        loss_vals = []
        for epoch in range(cfg.eval.epochs):
            train_time = time.time()

            for step, batch in enumerate(task["trainloader"]):
                # Heavy lifting is moved to engines
                device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask"])
                loss = model_engine.step(device_batch)
                loss_vals.append(loss.detach())
                if cfg.dryrun:
                    break

            metrics[task_name] = validate(model_engine, task["validloader"], metric, setup, cfg)
            stats[f"{task_name}_epoch"] += [epoch]
            stats[f"{task_name}_loss"] += [loss.item()]

            stats[f"{task_name}_avg_loss"] += [torch.stack(loss_vals).mean().item()]  # Smoothed loss
            loss_vals = []
            current_lr = model_engine.optimizer.param_groups[0]["lr"]

            log_msg = f"Train loss {loss.item():2.4f} at step {step} with lr {current_lr:.5f}. "
            log_msg += f"[Avg: {stats[f'{task_name}_avg_loss'][-1]:2.4f}] after epoch {epoch}."

            stats[f"{task_name}_train_time"] += [(time.time() - train_time)]
            estimated_train_finish = str(datetime.timedelta(seconds=stats[f"{task_name}_train_time"][-1] * cfg.eval.epochs))
            tokens_per_second = (step + 1) * cfg.eval.max_seq_length * cfg.impl.microbatch_size / stats[f"{task_name}_train_time"][-1]
            log_msg += (
                f" Perf: {stats[f'{task_name}_train_time'][-1]/60:2.4f}min per epoch ({tokens_per_second:.0f}t/s). "
                f"Estimated Total Train: {estimated_train_finish}."
            )

            for name, metric_val in metrics[task_name].items():
                stats[f"{task_name}_{name}"] += [metric_val]
            log.info(log_msg)
            msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in metrics[task_name].items()])
            log.info(f"Validation metric is {msg_metrics} after epoch {epoch}.")
            cramming.utils.wandb_log(stats, cfg)

            if cfg.dryrun:
                break
        # Launch testing:
        if task["extra_validloader"] is not None:
            extra_eval_metric = validate(model_engine, task["extra_validloader"], metric, setup, cfg)
            metrics[task_name + "extra"] = extra_eval_metric
            for name, metric_val in extra_eval_metric.items():
                stats[f"{task_name}_{name}_extra"] += [metric_val]
            msg_metrics = " ".join([f"{k}: {v:2.4f}" for k, v in extra_eval_metric.items()])
            log.info(f"Extra validation metric is {msg_metrics} after finetuning.")
            cramming.utils.wandb_log({f"{task_name}_{name}_extra": [v] for k, v in extra_eval_metric.items()}, cfg)

    # Save to summary:
    if cramming.utils.is_main_process():
        cramming.utils.dump_metrics(cfg, metrics)
        cramming.utils.save_summary("downstream", cfg, metrics, stats, time.time() - local_time, setup)


@torch.no_grad()
def validate(model_engine, validloader, metric, setup, cfg):
    """Evaluate on validation set."""
    model_engine.eval()
    for step, batch in enumerate(validloader):
        device_batch = model_engine.to_device(batch, keys=["input_ids", "labels", "attention_mask"])
        _, predictions = model_engine.forward_inference(**device_batch)
        metric.add_batch(predictions=predictions, references=device_batch["labels"])
        if cfg.dryrun and step > 1:
            break
    eval_metric = metric.compute()
    model_engine.train()
    return eval_metric


@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_downstream_process, job_name="downstream finetuning")


if __name__ == "__main__":
    launch()
