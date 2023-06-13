"""Prepare downstream tasks evaluations."""
import torch
import datasets

import os
import logging

from collections import defaultdict
from datasets import load_dataset

from .pretraining_preparation import main_process_first
from ..backend.utils import prepare_downstream_dataloader

log = logging.getLogger(__name__)


def prepare_task_dataloaders(tokenizer, cfg_eval, cfg_impl):
    """Load all datasets in eval.tasks for finetuning and testing."""
    cfg_eval.path = os.path.expanduser(cfg_eval.path)
    datasets.disable_caching()  # Could cache these
    max_seq_length = cfg_eval.max_seq_length
    tasks = defaultdict(dict)

    for task_name, task_details in cfg_eval.tasks.items():
        log.info(f"Preparing data for task {task_details.collection}-{task_name}.")
        tasks[task_name]["details"] = task_details
        raw_datasets = load_dataset(task_details.collection, task_name, cache_dir=cfg_impl.path)
        if "train_data_source" in task_details:  # some superGLUE tasks do not include train data
            raw_data_train = load_dataset(task_details.collection, task_details.train_data_source, cache_dir=cfg_impl.path)
            if cfg_eval.tasks["rte"].structure != task_details.structure:
                for new_name, old_name in zip(task_details.structure, cfg_eval.tasks["rte"].structure):
                    raw_data_train = raw_data_train.rename_column(old_name, new_name)
            raw_datasets["train"] = raw_data_train["train"]
            raw_datasets["validation"] = raw_datasets["test"]
        if not task_details.regression:
            if "label" in raw_datasets["train"].features:
                label_list = raw_datasets["train"].features["label"].names
                tasks[task_name]["num_classes"] = len(label_list)
                log.info(f"{task_name} has classes {label_list}.")
            elif "answer" in raw_datasets["train"].features:
                label_list = sorted(list(set(raw_datasets["train"]["answer"])))
                tasks[task_name]["num_classes"] = len(label_list)
                log.info(f"{task_name} has classes {label_list}.")
            else:
                label_list = None
                tasks[task_name]["num_classes"] = tokenizer.vocab_size
                log.info(f"{task_name} predicts a target token.")
        else:
            tasks[task_name]["num_classes"] = 1
            label_list = None

        def preprocess_function(examples):
            # Tokenize the texts
            if len(task_details.structure) == 2:
                texts = tuple(examples[sentence_key] for sentence_key in task_details.structure)
            elif len(task_details.structure) == 1:  # fake 2nd option all of the time, because this is the only way to get hf to use the
                # 2nd template option in the PostProcessor for the tokenizer
                # otherwise CoLA and SST2 won't have [CLS] tags applied correctly in all tokenizer settings
                main_texts = examples[task_details.structure[0]]
                fake_texts = ["" for example in examples[task_details.structure[0]]]
                texts = (main_texts, fake_texts)
            else:  # Merge first n blocks (like question, options) and then merge multiple answers
                texts = tuple(examples[sentence_key] for sentence_key in task_details.structure)
                premises = [tokenizer.sep_token.join([str(f) for f in fragments]) for fragments in zip(*texts[:-1])]  # join premises
                if isinstance(texts[-1][0], list):
                    # merge multiple hypotheses
                    hypothesis = [tokenizer.sep_token.join([f"{l}:{a}" for l, a in zip(label_list, answers)]) for answers in texts[-1]]
                else:
                    hypothesis = texts[-1]
                texts = (premises, hypothesis)

            result = tokenizer(
                *texts,
                max_length=max_seq_length,
                truncation=True,  # will cut off context if seq_length is too short!
                pad_to_multiple_of=cfg_impl.pad_to_multiple_of,
            )

            if "label" in examples:
                result["labels"] = examples["label"]
            elif "answer" in examples:
                result["labels"] = [[answer == l for l in label_list].index(True) for answer in examples["answer"]]
            elif "answers" in examples:  # this is RECORD-specific stuff
                # This is code to for RECORD into a simple classification problem and remove any feature engineering of the NLP task
                # This is unlikely to be the optimal way to solve record, but fits our evaluation that is focused solely on
                # only dumb downstream classification as a "sane" metric
                answer_ids = tokenizer([answer[0] for answer in examples["answers"]], max_length=3, truncation=True)["input_ids"]
                result["labels"] = [answer_id[0] for answer_id in answer_ids]
            else:
                raise ValueError("Could not find labels in dataset.")

            return result

        assert cfg_eval.evaluation_set in ["validation", "test"]
        raw_datasets.pop("test") if (cfg_eval.evaluation_set != "test" and "test" in raw_datasets) else None
        with main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                batch_size=1024,
                load_from_cache_file=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )

        train_dataset = processed_datasets["train"]
        train_dataset.set_format("torch")
        eval_dataset = processed_datasets[f"{cfg_eval.evaluation_set}_matched" if task_name == "mnli" else cfg_eval.evaluation_set]

        eval_dataset.set_format("torch")
        if task_name == "mnli":
            # Extra task loader for MNLI
            extra_eval_dataset = processed_datasets[f"{cfg_eval.evaluation_set}_mismatched"]
            extra_eval_dataset.set_format("torch")
        else:
            extra_eval_dataset = None

        train_dataloader, eval_dataloader, extra_eval_dataloader = _build_dataloaders(
            tokenizer,
            train_dataset,
            eval_dataset,
            extra_eval_dataset,
            cfg_impl,
        )

        tasks[task_name]["trainloader"] = train_dataloader
        tasks[task_name]["validloader"] = eval_dataloader
        tasks[task_name]["extra_validloader"] = extra_eval_dataloader

        if task_name == "multirc":  # special rule for this superGLUE task, carry idx on validation
            eval_dataloader.index_lookup = dict(zip(range(len(raw_datasets["validation"])), raw_datasets["validation"]["idx"]))

        # Log overviews so we always know what's going on with weird tokenization tricks
        random_sentence_idx = torch.randint(0, len(train_dataset), (1,)).item()
        input_data = train_dataset[random_sentence_idx]["input_ids"].squeeze()  # squeeze because hf has leading dim

        log.info(f"Random sentence with seq_length {tokenizer.model_max_length} from trainset of size {len(train_dataset):,}: ...")
        log.info(tokenizer.batch_decode(input_data[None])[0])
        log.info("... is tokenized into ...")
        log.info("_".join(tokenizer.decode(t) for t in input_data))
        if label_list is not None:
            log.info(f"Correct Answer: {label_list[train_dataset[random_sentence_idx]['labels']]}")
        else:
            log.info(f"Correct Answer: {train_dataset[random_sentence_idx]['labels']}")
        random_sentence_idx = torch.randint(0, len(eval_dataset), (1,)).item()
        input_data = eval_dataset[random_sentence_idx]["input_ids"].squeeze()  # squeeze because hf has leading dim

        log.info(f"Random sentence from validset of size {len(eval_dataset):,}: ...")
        log.info(tokenizer.batch_decode(input_data[None])[0])
        if label_list is not None:
            log.info(f"Correct Answer: {label_list[eval_dataset[random_sentence_idx]['labels']]}")
        else:
            log.info(f"Correct Answer: {eval_dataset[random_sentence_idx]['labels']}")

    return tasks


def _build_dataloaders(tokenizer, train_dataset, eval_dataset, extra_eval_dataset, cfg_impl):
    """Construct dataloaders according to cfg_impl settings. Validation samplers always repeat on all devices."""
    train_dataloader = prepare_downstream_dataloader(train_dataset, tokenizer, "training", cfg_impl)
    eval_dataloader = prepare_downstream_dataloader(eval_dataset, tokenizer, "eval", cfg_impl)
    if extra_eval_dataset is not None:
        extra_eval_dataloader = prepare_downstream_dataloader(extra_eval_dataset, tokenizer, "eval", cfg_impl)
    else:
        extra_eval_dataloader = None
    return train_dataloader, eval_dataloader, extra_eval_dataloader
