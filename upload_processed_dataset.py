"""Script to upload a processed dataset to the huggingface hub. You probably don't need this :)"""


import hydra
import logging
from omegaconf import OmegaConf
import tempfile
import os

from datasets import load_dataset

import cramming


log = logging.getLogger(__name__)


def upload(cfg, setup):
    dataset, tokenizer = cramming.load_pretraining_corpus(cfg.data, cfg.impl)
    checksum = cramming.data.utils.checksum_config(cfg.data)
    processed_dataset_name = f"{cfg.data.name}_{checksum}"

    use_own_chunking = True
    chunk_size = 8192 * 32
    num_files = len(dataset) // chunk_size + 1
    target_types = ["input_ids"]

    files = []
    # Split dataset in parquet files
    with tempfile.TemporaryDirectory() as tmpdirname:
        if use_own_chunking:
            # Loop through the dataset and write each chunk to a Parquet file
            # This is not really necessary, but nice to save only target_types and to match chunk sizes to target batch sizes
            for idx in range(num_files):
                chunk = dataset.select(range(idx * chunk_size, min(len(dataset), (idx + 1) * chunk_size)))
                filename = f"{tmpdirname}/train_{idx}.parquet"
                chunk.to_pandas()[target_types].to_parquet(filename, index=False)
                files.append(filename)
                log.info(f"Chunk {idx} written to file {filename}.")

            # Re-assemble parqueted dataset
            dataset = load_dataset("parquet", data_files=files)

        # Define the dataset info
        description = f"""This is a preprocessed dataset for the cramming-project.

                                Use only with the tokenizer prescribed here.
                                This version is {processed_dataset_name}, which corresponds to the following setup:
                                {OmegaConf.to_yaml(cfg, resolve=True)}

                                Limitations and bias:
                                This training data was further filtered and sorted beyond the normal preprocessing.
                                These modifications were not tested for unintended consequences.

                              """
        dataset["train"].info.description = description
        # dataset_tags = ["cramming", "English", "preprocessed"]

        # Launch upload
        log.info("Preparing for dataset upload ...")
        dataset.push_to_hub(processed_dataset_name, private=True)

        # Upload tokenizer to same adress - this is annoying because by default tokenizers are pushed to model directories
        # tokenizer.push_to_hub(processed_dataset_name) -> this will push to a new directory in HF models
        from huggingface_hub import HfApi

        api = HfApi()
        log.info("Preparing for tokenizer upload ...")
        tokenizer_loc = os.path.join(os.path.join(cfg.impl.path, processed_dataset_name), "tokenizer")
        for file in os.listdir(tokenizer_loc):
            api.upload_file(
                path_or_fileobj=os.path.join(tokenizer_loc, file),
                path_in_repo=os.path.join("tokenizer", file),
                repo_id=f"{api.whoami()['name']}/{processed_dataset_name}",
                repo_type="dataset",
            )
        log.info("Upload completed succesfully.")


@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, upload, job_name="upload")


if __name__ == "__main__":
    launch()
