# Cramming Language Model (Pretraining)

This repository contains code to replicate our research described in "Cramming: Training a Language Model on a Single GPU in One Day". We experiment with language model pretraining a BERT-type model with limited compute, wondering "how bad can it really be"?


You can find our paper here: https://arxiv.org/abs/2212.14034, and the abstract below:

> Recent trends in language modeling have focused on increasing performance through scaling, and have resulted in an environment where training language models is out of reach for most researchers and practitioners.  While most in the community are asking how to push the limits of extreme computation, we ask the opposite question:  
How far can we get with a single GPU in just one day?

> We investigate the downstream performance achievable with a transformer-based language model trained completely from scratch with masked language modeling for a *single* day on a *single consumer* GPU.
Aside from re-analyzing nearly all components of the pretraining pipeline for this scenario and providing a modified pipeline with performance close to BERT, we investigate why scaling down is hard, and which modifications actually improve performance in this scenario. We provide evidence that even in this constrained setting, performance closely follows scaling laws observed in large-compute settings. Through the lens of scaling laws, we categorize a range of recent improvements to training and architecture and discuss their merit and practical applicability (or lack thereof) for the limited compute setting.

## The Rules for Cramming
Setting:
* A transformer-based language model of arbitrary size is trained with masked-language modeling, completely from scratch.
* Existing pretrained models cannot be included in any part of the pipeline.
* Any raw text (excluding downstream data) can be included for training. This means that one can achieve speedups by making judicious choices about how and when to sample data, provided the sampling mechanism does not require a pre-trained model.
* The downloading and pre-processing of raw data is exempted from the total compute budget. Pre-processing may include CPU-based tokenizer construction, tokenization, and filtering, but cannot include representation learning (e.g. pre-training a word embedding is not allowed, unless it is counted towards the final runtime).
* Training proceeds on a single GPU for 24 hours.
* Downstream performance is evaluated on GLUE (https://gluebenchmark.com/). Downstream finetuning on GLUE is limited to brief training with only the training data of the downstream task (we consider 5 epochs or less) and needs to work with hyperparameters set globally for all GLUE tasks. Downstream finetuning is excluded from the total compute budget.


# How to run the code

## Requirements:
* PyTorch: `torch` (at least version 1.12)
* huggingface: `transformers`, `tokenizers`, `datasets`
* `hydra-core`
* [OPTIONAL]`deepspeed`
* [OPTIONAL] `flash-attention`
* `psutil`
* `einops`
* [OPTIONAL] For The-Pile data, install `zstandard`

## Installation
* Just clone for now, and install packages as described. Inside the cloned directory, you can use `pip install .` to install all packages and scripts.
* [Optional] For deduplication (necessary for the final dataset recipe), first install rust `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh `, then
`git clone https://github.com/google-research/deduplicate-text-datasets/tree/dev-v1` and then run `cargo install --target-dir ../cramming/dedup`
* [Optional] For FlashAttention (necessary for the `c5` recipe), install package as instructed at https://github.com/HazyResearch/flash-attention
* [Optional] Follow the instructions at https://pre-commit.com/ to install the pre-commit hooks (necessary only if you want to contribute to this project).

To verify a minimal installation, you can run
```
python pretrain.py name=test arch=bert-base train=bert-base data=sanity-check-2 dryrun=True impl.microbatch_size=2
```
This command pre-processes a small sanity-check dataset, and runs a single training step.


## General Usage

Use the `pretrain.py` script to pretrain with limited compute. This repository uses hydra (https://hydra.cc/docs/intro/), so all fields in `cramming/config` can be modified on the command line. For example, the `budget` can be modified by providing `budget=48` as additional argument, or the learning rate can be modified via `train.optim.lr=1e-4`. Check out the configuration folder to see all arguments.

Your first step should be to verify the installed packages. To do so, you can run `python pretrain.py dryrun=True`, which will run the default sanity check for a single iteration. From there, you can enable additional functionality. For example, modify the architecture, e.g. `arch=bert-original` and training setup `train=bert-original`.
To really train a language model, you need to switch away from the sanity check dataset to at least `data=bookcorpus-wikipedia`.

### Data Handling
The data sources from `data.sources` will be read, normalized and pretokenized before training starts and cached into a database. Subsequent calls with the same configuration will reused this database of tokenized sequences. By default, a new tokenizer will also be constructed and saved during this process. Important data options are `data.max_entries_in_raw_dataset`, which defines how much *raw* data will be loaded. For example, for a large data source such as C4, only a subset of raw data will be downloaded. Then, `max_seq_in_tokenized_dataset` bottlenecks how many *processed* sequences will be stored in the database. This number should be larger than the number of sequences expected to be read within the budget.

Additional Notes:
* A simple trick to run dataset preprocessing only is to run `python pretrain.py data=... dryrun=True`, which dry-runs the training, but runs the full data preprocessing. Later runs can then re-use the cached data.
* Dataset preprocessing is heavily parallelized. This might be a problem for your RAM. If this happens, reduce `impl.threads`. Especially the deduplication code does require substantial amounts of RAM.
* I would run first experiments with `bookcorpus-wikipedia` only, which preprocesses comparatively quickly and only then look into the full processed and filtered C4.


#### Preprocessed Datasets

For reference and if you are only interested in changing training/architecture, you can find some preprocessed datasets here:

https://www.dropbox.com/sh/sy8wanplx5typ9k/AAAWUceTcvZIh1GFX4Ij7_xXa?dl=0.

You will not need to download all of these. `c4-subset_WordPiecex32768_e0501aeb87699de7500dc91a54939f44` here is the final processed dataset for `data=c4-subset-processed` and `bookcorpus-wikitext_WordPiecex32768_a295a1f5b033756b08d2dbca690655a7` is the default `bookcorpus-wikipedia` dataset. Each folder contains a file called `model_config.json` that describes the preprocessing. You need to move these datasets into the `data` folder of your base output directory (so, `cramming/outputs/data` with default settings). The preprocessed data will be read and the preprocessing step skipped entirely, if the data is placed in the right folder.

Preprocessed data is convenient to work with, and I do think modifications to data processing and filtering continue to be under-explored compared to training and architecture because of this. There might be more gains to be had with better data, than with other tweaks, so ultimately you might want to consider setting up the code and environment for the full data processing pipeline to work.


### Evaluation

To evaluate pretrained models on GLUE (or some GLUE tasks), use `eval.py`. This script searches for saved models in the base directory. Given the name of a previous run, this script will, by default, retrieve the latest checkpoint saved with this name, and then run evaluations.


### WandB

You can log runs to your weights&biases account. To do so, simply modify `wandb.entity` and `wandb.project` on the command line or at `cramming/config/wandb/default.yaml`.



## Replicate the final recipe

To replicate the final recipe discussed in the paper, run
```
python pretrain.py name=amp_b4096_c5_o3_final arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset-processed
```
to pretrain and
```
python eval.py eval=GLUE_sane name=amp_b4096_c5_o3_final eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True
```
to evaluate the model. The recipe called "crammed BERT" in the paper corresponds to the architecture called `bert-c5` trained with training setup `bert-o3` on data `c4-subset-processed`.

## Additional Recipes
Pretraining:
Single GPU:
```
python pretrain.py name=bert data=bookcorpus-wikipedia arch=bert-original train=bert-original
```
Multi-GPU:
```
torchrun --nproc_per_node=4 --standalone pretrain.py name=bert4gpu  data=bookcorpus-wikipedia arch=bert-original train=bert-original
```

Eval a huggingface checkpoint:
```
python eval.py dryrun=True eval=rte name=bert-finetuning eval.checkpoint=hf://bert-base-uncased
```

Sanity check for distributed code on CPU:
```
torchrun --nproc_per_node=4 --standalone  pretrain.py name=speedtest1 dryrun=True data=sanity-check-2  impl.backend=gloo
```

Additional examples for recipes can be found in the `/scripts` folder.


# Todos:

The following options are currently broken/limited/work-in-progress. Use these at your own discretion. Of course, any contributions here are highly appreciated. You can also message me with more questions about any of these points, if you want to look into them.

* The-Pile needs to be downloaded in its entirety to be used, but the code could be updated to stream, just like C4.
* Data Preprocessing is wasteful in terms of RAM.
* Token Dropping is simplistic, a more involved version could be better.
* Code currently uses the "old" `jit.script` fusion, should move toward new `torch.compile` implementation at some point. The current `inductor` hook is also non-functional.
* Shampoo (see discussion at https://twitter.com/_arohan_/status/1608577721818546176?s=20)
* Causal Attention [I broke this shortly before release, if you want to re-test CA, you'd have to fix it first]
* LAWA

# Contact

Please, feel free to contact us with any questions, or open an issue on Github.
