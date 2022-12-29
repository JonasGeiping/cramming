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
* PyTorch: `torch`
* huggingface: `transformers`, `tokenizers`, `datasets`
* `hydra-core`
* [OPTIONAL]`deepspeed`
* [OPTIONAL] `flash-attention`
* `psutil`
* `einops`
* [OPTIONAL] For The-Pile data, install `zstandard`

## Installation
* Just clone for now, and install packages as described
* [Optional] Follow the instructions at https://pre-commit.com/ to install the pre-commit hooks.
* [Optional] For deduplication, first install rust `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh `, then
`git clone https://github.com/google-research/deduplicate-text-datasets/tree/dev-v1` and then run `cargo install --target-dir ../cramming/dedup`
* [Optional] For FlashAttention, install package as instructed at https://github.com/HazyResearch/flash-attention

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

# Contact

Please, feel free to contact us with any questions, or open an issue on github.
