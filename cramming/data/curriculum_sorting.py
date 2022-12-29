"""Baseline curricula."""
import torch
import numpy as np

import logging

log = logging.getLogger(__name__)


def _sort_tokenized_dataset_by_unigram(tokenized_dataset, tokenizer, num_threads=1, ngram=1, reverse=False):
    # Force unigram counts per token:
    map_setup = dict(
        batched=True,
        batch_size=1024,
        # num_proc=None,  # have to reimplement counting as in-out instead of side effects for this to work. Lets see how slow num_proc=0 is
        load_from_cache_file=False,
        # keep_in_memory=True,
    )

    unigrams_counts_per_token = np.zeros(tokenizer.vocab_size, dtype=np.int64)

    def count_unigrams(examples):
        nonlocal unigrams_counts_per_token
        unigrams_counts_per_token += np.bincount(np.asarray(examples["input_ids"]).reshape(-1), minlength=tokenizer.vocab_size)

    tokenized_dataset.map(count_unigrams, desc="Counting token unigrams", **map_setup, num_proc=None)

    token_count = sum(unigrams_counts_per_token)
    k = 1
    k_smoothed_probs = (unigrams_counts_per_token + k) / (token_count + k * tokenizer.vocab_size)
    log2_probs = np.log2(k_smoothed_probs)

    def return_seq_prob(examples):
        # seq_counts = np.apply_along_axis(np.bincount, axis=1, arr=np.asarray(examples["input_ids"]), minlength=tokenizer.vocab_size)
        # seq_counts = (np.asarray(examples["input_ids"])[:, :,None] == np.arange(0, tokenizer.vocab_size)[None, None, :]).sum(axis=1)  # slower so far
        # logprob_scores = (log2_probs * seq_counts).sum(axis=1) / tokenizer.model_max_length
        # why make hard when can do easy?
        logprob_scores = log2_probs[np.asarray(examples["input_ids"])].sum(axis=1) / tokenizer.model_max_length
        return dict(scores=logprob_scores)

    dataset_probs = tokenized_dataset.map(
        return_seq_prob,
        desc="Computing log probs per sequence",
        remove_columns=tokenized_dataset.column_names,
        **map_setup,
        num_proc=num_threads if num_threads > 0 else None,
    )

    new_order = np.argsort(np.asarray(dataset_probs["scores"]))

    if reverse:
        new_order = new_order[::-1]

    return tokenized_dataset.select(indices=new_order, writer_batch_size=1024)


def _sort_tokenized_dataset_by_token(tokenized_dataset, tokenizer, target_token_id, num_threads=1):
    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=num_threads if num_threads > 0 else None,
        load_from_cache_file=False,
        # keep_in_memory=True,
    )

    def count_token(examples):
        return dict(counts=(np.asarray(examples["input_ids"]) == target_token_id).sum(axis=1))

    dataset_counts = tokenized_dataset.map(
        count_token,
        desc=f"Counting occurences of token {tokenizer.decode(target_token_id)}",
        remove_columns=tokenized_dataset.column_names,
        **map_setup,
    )

    new_order = np.argsort(np.asarray(dataset_counts["counts"]))[::-1]

    # Print sentence with most occurences:
    sentence_idx = int(new_order[0])
    input_data = torch.as_tensor(tokenized_dataset[sentence_idx]["input_ids"]).squeeze()  # squeeze because hf has leading dim
    dataset_size = len(tokenized_dataset)

    log.info("Sentence with most occurences of token ...")
    log.info(tokenizer.batch_decode(input_data[None])[0])

    sentence_idx = int(new_order[-1])
    input_data = torch.as_tensor(tokenized_dataset[sentence_idx]["input_ids"]).squeeze()  # squeeze because hf has leading dim
    dataset_size = len(tokenized_dataset)

    log.info("Sentence with least occurences of token ...")
    log.info(tokenizer.batch_decode(input_data[None])[0])

    return tokenized_dataset.select(indices=new_order, writer_batch_size=1024)


def _sort_tokenized_dataset_by_word_length(tokenized_dataset, tokenizer, num_threads=1):
    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=num_threads if num_threads > 0 else None,
        load_from_cache_file=False,
        # keep_in_memory=True,
    )

    def count_word_lengths(examples):
        return dict(lengths=[len(s) for s in tokenizer.batch_decode(torch.as_tensor(examples["input_ids"]))])

    dataset_counts = tokenized_dataset.map(
        count_word_lengths,
        desc="Counting word lengths per sequence",
        remove_columns=tokenized_dataset.column_names,
        **map_setup,
    )

    new_order = np.argsort(np.asarray(dataset_counts["lengths"]))  # shortest sentences first

    # Print sentence with shortest length
    sentence_idx = int(new_order[0])
    input_data = torch.as_tensor(tokenized_dataset[sentence_idx]["input_ids"]).squeeze()  # squeeze because hf has leading dim
    dataset_size = len(tokenized_dataset)

    log.info("Sentence with shortest length ...")
    log.info(tokenizer.batch_decode(input_data[None])[0])

    sentence_idx = int(new_order[-1])
    input_data = torch.as_tensor(tokenized_dataset[sentence_idx]["input_ids"]).squeeze()  # squeeze because hf has leading dim
    dataset_size = len(tokenized_dataset)

    log.info("and longest ...")
    log.info(tokenizer.batch_decode(input_data[None])[0])

    return tokenized_dataset.select(indices=new_order, writer_batch_size=1024)
