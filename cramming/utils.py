"""System utilities."""

import socket
import sys

import os
import csv
import yaml
import psutil
import multiprocess  # hf uses this for some reason
import collections

import torch
import transformers


import json
import random
import numpy as np
import time
import datetime
import tempfile
from .data.utils import checksum_config

import logging
import hydra
from omegaconf import OmegaConf, open_dict

log = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "0"


def main_launcher(cfg, main_fn, job_name=""):
    """This is boiler-plate code for a launcher."""
    launch_time = time.time()
    # Set definitive random seed:
    if cfg.seed is None:
        cfg.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    # Figure out all paths:
    with open_dict(cfg):
        cfg.original_cwd = hydra.utils.get_original_cwd()
        # ugliest way to get the absolute path to output subdir
        if not os.path.isabs(cfg.base_dir):
            base_dir_full_path = os.path.abspath(os.getcwd())
            while os.path.basename(base_dir_full_path) != cfg.base_dir:
                base_dir_full_path = os.path.dirname(base_dir_full_path)
                if base_dir_full_path == "/":
                    raise ValueError("Cannot find base directory.")
            cfg.base_dir = base_dir_full_path

        cfg.impl.path = os.path.expanduser(cfg.impl.path)
        if not os.path.isabs(cfg.impl.path):
            cfg.impl.path = os.path.join(cfg.base_dir, cfg.impl.path)

    # Decide GPU and possibly connect to distributed setup
    setup = system_startup(cfg)
    # Initialize wanDB :>
    if cfg.wandb.enabled:
        _initialize_wandb(setup, cfg)
    log.info("--------------------------------------------------------------")
    log.info(f"--------------Launching {job_name} run! ---------------------")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    main_fn(cfg, setup)

    log.info("-------------------------------------------------------------")
    log.info(f"Finished running job {cfg.name} with total train time: " f"{str(datetime.timedelta(seconds=time.time() - launch_time))}")
    if torch.cuda.is_available():
        max_alloc = f"{torch.cuda.max_memory_allocated(setup['device'])/float(1024**3):,.3f} GB"
        max_reserved = f"{torch.cuda.max_memory_reserved(setup['device'])/float(1024**3):,.3f} GB"
        log.info(f"Max. Mem allocated: {max_alloc}. Max. Mem reserved: {max_reserved}.")
    log.info("-----------------Job finished.-------------------------------")


def system_startup(cfg):
    """Decide and print GPU / CPU / hostname info. Generate local distributed setting if running in distr. mode.

    Set all required and interesting environment variables.
    """
    torch.backends.cudnn.benchmark = cfg.impl.benchmark
    torch.multiprocessing.set_sharing_strategy(cfg.impl.sharing_strategy)

    if cfg.impl.tf32_allowed:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway

    if cfg.impl.no_jit_compilation:
        torch.jit._state.disable()
    else:
        if torch.cuda.is_available():
            set_jit_instructions(cfg.impl.jit_instruction_type)
        else:
            set_jit_instructions("default")
    multiprocess.set_start_method("forkserver")
    if cfg.impl.local_staging_dir is not None:
        tmp_path = os.path.join(cfg.impl.local_staging_dir, "tmp")
        os.makedirs(tmp_path, exist_ok=True)
        os.environ["TMPDIR"] = tmp_path
        tempfile.tempdir = None  # Force temporary directory regeneration
    if cfg.impl.enable_huggingface_offline_mode:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # datasets will automatically disable tokenizer parallelism when needed:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_RS_NUM_CPUS"] = str(min(torch.get_num_threads(), cfg.impl.threads))
    max_dataset_memory = f"{psutil.virtual_memory().total // 2 // max(torch.cuda.device_count(), 1)}"
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = max_dataset_memory

    if not torch.cuda.is_available() and not cfg.dryrun:
        raise ValueError(
            f"No GPU allocated to this process on {socket.gethostname()} with name {cfg.name}. Running in CPU-mode is likely an error."
        )

    # Force thread reduction for all cases:
    torch.set_num_threads(min(torch.get_num_threads(), cfg.impl.threads))

    # Distributed launch?
    if "LOCAL_RANK" in os.environ:
        threads_per_gpu = min(torch.get_num_threads() // max(1, torch.cuda.device_count()), cfg.impl.threads)
        os.environ["OMP_NUM_THREADS"] = str(threads_per_gpu)
        local_rank = int(os.environ["LOCAL_RANK"])
        cfg.impl.local_rank = local_rank
        torch.distributed.init_process_group(backend=cfg.impl.backend)
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        run = os.environ.get("TORCHELASTIC_RUN_ID", "unknown")
        log.info(
            f"Distributed worker initialized on rank {global_rank} (local rank {local_rank}) "
            f"with {world_size} total processes. Run ID is {run}."
        )
        log.setLevel(logging.INFO if is_main_process() else logging.ERROR)
    else:
        os.environ["OMP_NUM_THREADS"] = str(min(torch.get_num_threads(), cfg.impl.threads))
        global_rank = local_rank = 0
        cfg.impl.local_rank = local_rank

    # Construct setup dictionary:
    dtype = getattr(torch, cfg.impl.default_precision)  # :> dont mess this up
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        log.info(f"GPU : {torch.cuda.get_device_name(device=device)}")
    setup = dict(device=device, dtype=dtype)
    python_version = sys.version.split(" (")[0]

    if local_rank == 0:
        log.info(f"Platform: {sys.platform}, Python: {python_version}, PyTorch: {torch.__version__}")
        log.info(f"CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")

    # 100% reproducibility?
    if cfg.impl.deterministic:
        set_deterministic()
    if cfg.seed is not None:
        if is_main_process():
            log.info(f"Seeding with random seed {cfg.seed} on rank 0.")
        set_random_seed(cfg.seed + 10 * global_rank)

    return setup


def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def num_processes():
    num_procs = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    return num_procs


def find_pretrained_checkpoint(cfg, downstream_classes=None):
    """Load a checkpoint either locally or from the internet."""
    local_checkpoint_folder = os.path.join(cfg.base_dir, cfg.name, "checkpoints")
    if cfg.eval.checkpoint == "latest":
        # Load the latest local checkpoint
        all_checkpoints = [f for f in os.listdir(local_checkpoint_folder)]
        checkpoint_paths = [os.path.join(local_checkpoint_folder, c) for c in all_checkpoints]
        checkpoint_name = max(checkpoint_paths, key=os.path.getmtime)
    elif cfg.eval.checkpoint == "smallest":
        # Load maybe the local checkpoint with smallest loss
        all_checkpoints = [f for f in os.listdir(local_checkpoint_folder)]
        checkpoint_paths = [os.path.join(local_checkpoint_folder, c) for c in all_checkpoints]
        checkpoint_losses = [float(path[-5:]) for path in checkpoint_paths]
        checkpoint_name = checkpoint_paths[np.argmin(checkpoint_losses)]
    elif not os.path.isabs(cfg.eval.checkpoint) and not cfg.eval.checkpoint.startswith("hf://"):
        # Look locally for a checkpoint with this name
        checkpoint_name = os.path.join(local_checkpoint_folder, cfg.eval.checkpoint)
    elif cfg.eval.checkpoint.startswith("hf://"):
        # Download this checkpoint directly from huggingface
        model_name = cfg.eval.checkpoint.split("hf://")[1].removesuffix("-untrained")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        cfg_arch = transformers.AutoConfig.from_pretrained(model_name)
        model_file = cfg.eval.checkpoint
        checkpoint_name = None
    else:
        # Look for this name as an absolute path
        checkpoint_name = cfg.eval.checkpoint

    if checkpoint_name is not None:
        # Load these checkpoints locally, might not be a huggingface model
        tokenizer = tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_name)
        with open(os.path.join(checkpoint_name, "model_config.json"), "r") as file:
            cfg_arch = OmegaConf.create(json.load(file))  # Could have done pure hydra here, but wanted interop

        # Use merge from default config to build in new arguments
        # with hydra.initialize(config_path="config/arch"):
        # cfg_default = OmegaConf.load(os.path.join(cfg.original_cwd, "cramming/config/arch/bert-base.yaml"))
        # cfg_arch = OmegaConf.merge(cfg_default, cfg_arch)

        # Optionally modify parts of the arch at eval time. This is not guaranteed to be a good idea ...
        # All mismatched parameters will be randomly initialized ...
        if cfg.eval.arch_modifications is not None:
            cfg_arch = OmegaConf.merge(cfg_arch, cfg.eval.arch_modifications)
        model_file = os.path.join(checkpoint_name, "model.pth")

        print(cfg_arch)

    log.info(f"Loading from checkpoint {model_file}...")
    return tokenizer, cfg_arch, model_file


def save_summary(table_name, cfg, metrics, stats, local_time, setup, original_cwd=True):
    """Save two summary tables. A detailed table of iterations/loss+acc and a summary of the end results."""
    # 1) detailed table:
    for step in range(len(stats["loss"])):
        iteration = dict()
        for key in stats:
            iteration[key] = stats[key][step] if step < len(stats[key]) else None
        save_to_table(".", f"{cfg.name}_convergence_results", dryrun=cfg.dryrun, **iteration)

    def _maybe_record(key, step=-1):
        try:
            return stats[key][step]
        except (IndexError, ValueError):
            return ""

    if "data" in cfg:
        processed_dataset_dir = f"{cfg.data.name}_{checksum_config(cfg.data)}"
    else:
        processed_dataset_dir = None
    base_name = cfg.base_dir.rstrip(os.sep).split(os.sep)[-1]
    local_folder = os.getcwd().split(base_name)[1].lstrip(os.sep)

    # Add some compute metrics:
    metrics["GPU"] = torch.cuda.get_device_name(device=setup["device"]) if torch.cuda.device_count() > 0 else ""
    metrics["numGPUs"] = torch.cuda.device_count()
    metrics["VRAM"] = torch.cuda.max_memory_allocated(setup["device"]) / float(1 << 30)
    metrics["RAM"] = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    # Flatten metrics:
    metrics = flatten(metrics)
    # 2) save a reduced summary
    if table_name == "pretrain":
        summary = dict(
            name=cfg.name,
            budget=cfg.budget,
            dataset="_".join(processed_dataset_dir.split("_")[:-1]),
            backend=cfg.impl.name,
            arch=" ".join(cfg.arch.architectures),
            loss=_maybe_record("loss"),
            final_step=_maybe_record("step"),
            final_epoch=_maybe_record("epoch"),
            step_time=np.mean(stats["train_time"]) if len(stats["train_time"]) > 0 else "",
            loss100k=_maybe_record("loss", step=100_000 // cfg.impl.print_loss_every_nth_step),
            loss200k=_maybe_record("loss", step=200_000 // cfg.impl.print_loss_every_nth_step),
            loss300k=_maybe_record("loss", step=300_000 // cfg.impl.print_loss_every_nth_step),
            **{k: v for k, v in metrics.items()},
            total_time=str(datetime.timedelta(seconds=local_time)).replace(",", ""),
            batch_size=cfg.train.batch_size,
            lr=cfg.train.optim.lr,
            warmup=cfg.train.warmup_steps,
            steps=cfg.train.steps,
            # System settings:
            seed=cfg.seed,
            dataset_hash=processed_dataset_dir.split("_")[-1],
            base_dir=cfg.base_dir,
            impl_path=cfg.impl.path,
            local_folder=local_folder,
            # # Dump configs from here on:
            **{f"Data_{k}": v for k, v in cfg.data.items()},
            **{f"Arch_{k}": v for k, v in cfg.arch.items()},
            **{f"Train_{k}": v for k, v in cfg.train.items()},
        )
    else:
        summary = dict(
            name=cfg.name,
            backend=cfg.impl.name,
            checkpoint=cfg.eval.checkpoint,
            loss=_maybe_record("loss"),
            avg_loss=_maybe_record("avg_loss"),
            final_epoch=_maybe_record("epoch"),
            step_time=np.mean(stats["train_time"]) if len(stats["train_time"]) > 0 else "",
            **{k: v for k, v in metrics.items()},
            total_time=str(datetime.timedelta(seconds=local_time)).replace(",", ""),
            batch_size=cfg.eval.batch_size,
            lr=cfg.eval.optim.lr,
            warmup=cfg.eval.warmup_steps,
            # System settings:
            seed=cfg.seed,
            base_dir=cfg.base_dir,
            impl_path=cfg.impl.path,
            local_folder=local_folder,
            # # Dump configs from here on:
            **{f"Eval_{k}": v for k, v in cfg.eval.items()},
        )
    location = os.path.join(cfg.original_cwd, "tables") if original_cwd else "tables"
    save_to_table(location, f"{table_name}_reports", dryrun=cfg.dryrun, **summary)


def save_to_table(out_dir, table_name, dryrun, **kwargs):
    """Save keys to .csv files."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f"table_{table_name}.csv")
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)  # noqa  # this line is testing the header
            # assert header == fieldnames[:len(header)]  # new columns are ok, but old columns need to be consistent
            # dont test, always write when in doubt to prevent erroneous table deletions
    except Exception as e:  # noqa
        if not dryrun:
            # print('Creating a new .csv table...')
            with open(fname, "w") as f:
                writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
                writer.writeheader()
        else:
            pass

    # Write a new row
    if not dryrun:
        # Add row for this experiment
        with open(fname, "a") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            writer.writerow(kwargs)
    else:
        pass


def set_random_seed(seed=233):
    """."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    # Can't be too careful :>


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_jit_instructions(type="nvfuser"):
    """Refer also https://github.com/pytorch/pytorch/blob/c90be037b46f58d2b120f46a1c466976f66817b5/torch/jit/_fuser.py#L20"""
    # torch._C._set_graph_executor_optimize(True)
    if type == "nvfuser":
        # from BERT nvidia example
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_fusion_strategy([("STATIC", 20), ("DYNAMIC", 20)])  # maybe this is overkill
    elif type == "nvfuser-profiler":
        # via https://github.com/tunib-ai/oslo/blob/master/oslo/torch/jit/_utils.py
        torch._C._jit_set_nvfuser_enabled(True)  # fuser2
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_texpr_fuser_enabled(False)  # fuser1
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_fusion_strategy([("STATIC", 20), ("DYNAMIC", 20)])  # maybe this is overkill
        # torch._C._debug_set_autodiff_subgraph_inlining(False)
    elif type == "nnc":
        # via https://github.com/tunib-ai/oslo/blob/master/oslo/torch/jit/_utils.py
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
    elif type == "legacy":
        # via https://github.com/tunib-ai/oslo/blob/master/oslo/torch/jit/_utils.py
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
    else:
        # default options
        pass


def avg_n_dicts(dicts):
    """https://github.com/wronnyhuang/metapoison/blob/master/utils.py."""
    # given a list of dicts with the same exact schema, return a single dict with same schema whose values are the
    # key-wise average over all input dicts
    means = {}
    for dic in dicts:
        for key in dic:
            if key not in means:
                if isinstance(dic[key], list):
                    means[key] = [0 for entry in dic[key]]
                else:
                    means[key] = 0
            if isinstance(dic[key], list):
                for idx, entry in enumerate(dic[key]):
                    means[key][idx] += entry / len(dicts)
            else:
                means[key] += dic[key] / len(dicts)
    return means


def dump_metrics(cfg, metrics):
    """Simple yaml dump of metric values."""

    filepath = f"metrics_{cfg.name}.yaml"
    sanitized_metrics = dict()
    for metric, val in metrics.items():
        try:
            sanitized_metrics[metric] = np.asarray(val).item()
        except ValueError:
            sanitized_metrics[metric] = np.asarray(val).tolist()
    with open(filepath, "w") as yaml_file:
        yaml.dump(sanitized_metrics, yaml_file, default_flow_style=False)


def _initialize_wandb(setup, cfg):
    if is_main_process():
        import wandb

        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        settings = wandb.Settings(start_method="thread")
        settings.update({"git_root": cfg.original_cwd})
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            settings=settings,
            name=cfg.name,
            mode="disabled" if cfg.dryrun else None,
            tags=cfg.wandb.tags if len(cfg.wandb.tags) > 0 else None,
            config=config_dict,
        )
        run.summary["GPU"] = torch.cuda.get_device_name(device=setup["device"]) if torch.cuda.device_count() > 0 else ""
        run.summary["numGPUs"] = torch.cuda.device_count()


def wandb_log(stats, cfg):
    if cfg.wandb.enabled:
        if is_main_process():
            import wandb

            wandb.log({k: v[-1] for k, v in stats.items()}, step=stats["step"][-1] if "step" in stats else None)


def flatten(d, parent_key="", sep="_"):
    """Straight-up from https://stackoverflow.com/a/6027615/3775820."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
