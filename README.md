# Cramming Language Model (Pretraining)

This repository contains code to replicate our research described in "Cramming: Training a Language Model on a Single GPU in One Day". We experiment with language model pretraining a BERT-type model with limited compute, wondering "how bad can it really be"?


You can find our paper here: https://arxiv.org/abs/2212.14034, and the abstract below:

> Recent trends in language modeling have focused on increasing performance through scaling, and have resulted in an environment where training language models is out of reach for most researchers and practitioners.  While most in the community are asking how to push the limits of extreme computation, we ask the opposite question:  
How far can we get with a single GPU in just one day?

> We investigate the downstream performance achievable with a transformer-based language model trained completely from scratch with masked language modeling for a *single* day on a *single consumer* GPU.
Aside from re-analyzing nearly all components of the pretraining pipeline for this scenario and providing a modified pipeline with performance close to BERT, we investigate why scaling down is hard, and which modifications actually improve performance in this scenario. We provide evidence that even in this constrained setting, performance closely follows scaling laws observed in large-compute settings. Through the lens of scaling laws, we categorize a range of recent improvements to training and architecture and discuss their merit and practical applicability (or lack thereof) for the limited compute setting.


## UPDATE: This is the new version of the framework!

You need PyTorch 2.0 to run the new code. If you want to remain on PyTorch 1.*, you can checkout the tag `Last1.13release`. The new model, trained with the new codebase is 1-2% better on GLUE with the same budget. The checkpoint can be found at https://huggingface.co/JonasGeiping/crammed-bert. The old checkpoint is now https://huggingface.co/JonasGeiping/crammed-bert-legacy.

Also, data preprocessing has improved, you can now stream data directly from huggingface, from the upload at https://huggingface.co/datasets/JonasGeiping/the_pile_WordPiecex32768_2efdb9d060d1ae95faf952ec1a50f020.


## The Rules for Cramming
Setting:
* A transformer-based language model of arbitrary size is trained with masked-language modeling, completely from scratch.
* Existing pretrained models cannot be included in any part of the pipeline.
* Any raw text (excluding downstream data) can be included for training. This means that one can achieve speedups by making judicious choices about how and when to sample data, provided the sampling mechanism does not require a pre-trained model.
* The downloading and pre-processing of raw data is exempted from the total compute budget. Pre-processing may include CPU-based tokenizer construction, tokenization, and filtering, but cannot include representation learning (e.g. pre-training a word embedding is not allowed, unless it is counted towards the final runtime).
* Training proceeds on a single GPU for 24 hours.
* Downstream performance is evaluated on GLUE (https://gluebenchmark.com/). Downstream finetuning on GLUE is limited to brief training with only the training data of the downstream task (we consider 5 epochs or less) and needs to work with hyperparameters set globally for all GLUE tasks. Downstream finetuning is excluded from the total compute budget.


# How to run the code

Run  `pip install .` to install all dependencies.


## Requirements in Details:
* PyTorch: `torch` (at least version 2.1)
* huggingface: `transformers`, `tokenizers`, `datasets`, `evaluate`
* `hydra-core`
* `psutil`, `pynvml`, `safetensors`
* `einops`

## Installation
* Just clone for now, and install packages as described. Inside the cloned directory, you can use `pip install .` to install all packages and scripts.
* [Optional] For deduplication (only needed to replicate deduplication tests), first install rust `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh `, then
`git clone https://github.com/google-research/deduplicate-text-datasets/tree/dev-v1` and then run `cargo install --target-dir ../cramming/dedup`
* [Optional] Follow the instructions at https://pre-commit.com/ to install the pre-commit hooks (necessary only if you want to contribute to this project).

To verify a minimal installation, you can run
```
python pretrain.py name=test arch=hf-bert-base train=bert-base data=sanity-check-2 dryrun=True impl.microbatch_size=2
```
This command pre-processes a small sanity-check dataset, and runs a single training step.


## General Usage

Use the `pretrain.py` script to pretrain with limited compute. This repository uses hydra (https://hydra.cc/docs/intro/), so all fields in `cramming/config` can be modified on the command line. For example, the `budget` can be modified by providing `budget=48` as additional argument (to run for 48 hours), or the learning rate can be modified via `train.optim.lr=1e-4`. Check out the configuration folder to see all arguments.

Your first step should be to verify the installed packages. To do so, you can run `python pretrain.py dryrun=True`, which will run the default sanity check for a single iteration. From there, you can enable additional functionality. For example, modify the architecture, e.g. `arch=bert-original` and training setup `train=bert-original`.
To really train a language model, you need to switch away from the sanity check dataset to at least `data=pile-readymade`. Then, choose an improved training setup, e.g. `train=bert-o4`, and an improved model layout, e.g. `arch=crammed-bert`.

### Data Handling
The data sources from `data.sources` will be read, normalized and pretokenized before training starts and cached into a database. Subsequent calls with the same configuration will reused this database of tokenized sequences. By default, a new tokenizer will also be constructed and saved during this process. Important data options are `data.max_entries_in_raw_dataset`, which defines how much *raw* data will be loaded. For example, for a large data source such as C4, only a subset of raw data will be downloaded. Then, `max_seq_in_tokenized_dataset` bottlenecks how many *processed* sequences will be stored in the database. This number should be larger than the number of sequences expected to be read within the budget.

Additional Notes:
* Start with preprocessed data, using `data=pile-readymade`
* A simple trick to run dataset preprocessing only is to run `python pretrain.py data=... dryrun=True`, which dry-runs the training, but runs the full data preprocessing. Later runs can then re-use the cached data.
* Dataset preprocessing is heavily parallelized. This might be a problem for your RAM. If this happens, reduce `impl.threads`. Especially the deduplication code does require substantial amounts of RAM.
* I would run first data experiments with `bookcorpus-wikipedia` only, which preprocesses comparatively quickly and only then look into the full processed and filtered C4.


#### Preprocessed Datasets

For reference and if you are only interested in changing training/architecture, you can find some preprocessed datasets here:

* https://huggingface.co/datasets/JonasGeiping/the_pile_WordPiecex32768_2efdb9d060d1ae95faf952ec1a50f020
* https://huggingface.co/datasets/JonasGeiping/the_pile_WordPiecex32768_8eb2d0ea9da707676c81314c4ea04507
* https://huggingface.co/datasets/JonasGeiping/the_pile_WordPiecex32768_2efdb9d060d1ae95faf952ec1a50f020

These data sources can be streamed. To do so, simple set `data=pile-readymade`.

Preprocessed data is convenient to work with, and I do think modifications to data processing and filtering continue to be under-explored compared to training and architecture because of this. There might be more gains to be had with better data, than with other tweaks, so ultimately you might want to consider setting up the code and environment for the full data processing pipeline to work.


#### Model Checkpoint

You can now find a checkpoint for the final version trained on `the-pile` at https://huggingface.co/JonasGeiping/crammed-bert.

### Evaluation

To evaluate pretrained models on GLUE (or some GLUE tasks), use `eval.py`. This script searches for saved models in the base directory. Given the name of a previous run, this script will, by default, retrieve the latest checkpoint saved with this name, and then run evaluations.


### WandB

You can log runs to your weights&biases account. To do so, simply modify `wandb.entity` and `wandb.project` on the command line or at `cramming/config/wandb/default.yaml`.



## Replicate the final recipe

To replicate the final recipe discussed in the paper, run
```
python pretrain.py name=amp_b8192_cb_o4_final arch=crammed-bert train=bert-o4  data=pile-readymade
```
to pretrain and
```
python eval.py eval=GLUE_sane name=amp_b8192_cb_o4_final eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True impl.compile_torch=False
```
to evaluate the model. The recipe called "crammed BERT" in the paper corresponds to the architecture called `crammed-bert` in the config,  trained with the training setup `bert-o4` on data `the-pile`.

#### Inductor Settings

For optimal performance, you need to be on the latest pytorch nightly and set the following inductor variables (which modify the `torch.compile` setup using inductor):
* `max_autotune_gemm: True`
* `max_autotune_pointwise: False`
* `triton.cudagraphs: True`
* `triton.cudagraph_trees: False`

## Additional Recipes
Pretraining:
Single GPU, original BERT settings:
```
python pretrain.py name=bert data=bookcorpus-wikipedia arch=bert-original train=bert-original budget=10000000
```
Multi-GPU, original BERT settings:
```
torchrun --nproc_per_node=4 --standalone pretrain.py name=bert4gpu  data=bookcorpus-wikipedia arch=bert-original train=bert-original budget=10000000 impl.fullgraph=false impl._inductor_vars.triton.cudagraphs=False
```

Eval a huggingface checkpoint (in this example on RTE):
```
python eval.py eval=GLUE_sane eval/task=rte name=bert-finetuning eval.checkpoint=hf://bert-base-uncased impl.shuffle_in_dataloader=True impl.compile_torch=False impl.microbatch_size=16
```
Eval a local checkpoint (disable compilation, which expect fixed shapes right now):
```
python eval.py eval=GLUE_sane eval/task=rte name=NAME_OF_PRETRAINING_RUN eval.checkpoint=latest impl.shuffle_in_dataloader=True impl.compile_torch=False
```

Sanity check for distributed code on CPU:
```
CUDA_VISIBLE_DEVICES= torchrun --nproc_per_node=2 --standalone  pretrain.py name=cpu_multi_check dryrun=True data=sanity-check-2  impl.dist_backend=gloo impl.fullgraph=false impl._inductor_vars.triton.cudagraphs=False
```

Additional examples for recipes can be found in the `/scripts` folder.


# Todos:

The following options are currently broken/limited/work-in-progress. Use these at your own discretion. Of course, any contributions here are highly appreciated. You can also message me with more questions about any of these points, if you want to look into them.

* Shampoo (see discussion at https://twitter.com/_arohan_/status/1608577721818546176?s=20). In general, alternative optimizers are probably undertested, and could use more atttention.
* Some options are only available in the old release at the `Last1.13release` tag. If you are interested in reviving some of these options. Feel free to open a pull request with updates to the new codebase.

# Contact

Please, feel free to contact us with any questions, or open an issue on Github.
