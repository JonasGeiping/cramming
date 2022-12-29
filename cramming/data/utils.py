"""Various utilities."""
import os
from omegaconf import OmegaConf
import hashlib
import json
import shutil

import logging
import time

import datasets

log = logging.getLogger(__name__)


def checksum_config(cfg):
    """This is more annoying that I thought it would be. But a json-dump of the config file is hashed and used as checksum."""
    bindump = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True).encode("utf-8")
    checksum_of_config = hashlib.md5(bindump).hexdigest()
    if "tokenizer" in cfg and "vocab_size" in cfg:
        checksum_of_config = f"{cfg.tokenizer}x{cfg.vocab_size}_{checksum_of_config}"
    return checksum_of_config


def stage_dataset(data_directory_path, local_staging_dir):
    """This is a mess because our network drives are a mess. You might not need this."""
    data_directory_name = os.path.basename(data_directory_path)
    new_path = os.path.join(local_staging_dir, data_directory_name)
    if os.path.isdir(data_directory_path):
        try:
            if not os.path.isdir(new_path):
                try:
                    shutil.copytree(data_directory_path, new_path)
                    log.info(f"Staging dataset to {new_path}...")
                except FileExistsError:
                    log.info(f"Concurrent writing to {new_path} detected. Stopping staging in this run and waiting for 300 seconds.")
                    time.sleep(300)
            else:
                log.info(f"Using staged dataset found at {new_path}...")

            for retries in range(15):
                _, _, free = shutil.disk_usage(new_path)
                used = _get_size(new_path)
                try:
                    tokenized_dataset = datasets.load_from_disk(new_path)
                    log.info(f"Staged dataset size is {used / 1024**3:,.3f}GB. {free/ 1024**3:,.3f}GB free in staging dir.")
                    return new_path
                except FileNotFoundError:
                    log.info(
                        f"Staged dataset is incomplete. Size is {used / 1024**3:,.3f}GB. "
                        f" Waiting for 60 more secs for staging race condition."
                    )
                    time.sleep(60)
            log.info(f"Staging dataset corrupted. Falling back to network drive location {data_directory_path}")
            return data_directory_path

        except Exception as e:  # noqa
            log.info(f"Staging failed with error {e}. Falling back to network drive location {data_directory_path}")
            return data_directory_path
    else:
        raise FileNotFoundError(f"Dataset not yet generated or not found at {data_directory_path}.")


def _get_size(start_path="."):
    """Compute the size of a directory path. Why is this not in the standard library?"""
    """Stolen from https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size
