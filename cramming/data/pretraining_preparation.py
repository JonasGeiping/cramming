"""Prepare and preprocess datasets."""

import torch
import datasets
import hydra

import os
import contextlib
import logging
import tempfile
from itertools import chain
from collections import defaultdict

import json
from omegaconf import OmegaConf


from .tokenizer_preparation import construct_tokenizer, load_tokenizer
from .curriculum_sorting import _sort_tokenized_dataset_by_unigram, _sort_tokenized_dataset_by_token, _sort_tokenized_dataset_by_word_length
from .deduplicate import deduplicate_huggingface_dataset
from .utils import checksum_config, stage_dataset, detailed_OSError


log = logging.getLogger(__name__)
datasets.enable_progress_bar()
datasets.disable_caching()  # We'll save only the final preprocessed dataset


def load_pretraining_corpus(cfg_data, cfg_impl):
    """Load (and optionally stage) a pre-processed corpus. Create one if it doesn't exist."""
    datasets.disable_caching()
    checksum = checksum_config(cfg_data)

    processed_dataset_dir = f"{cfg_data.name}_{checksum}"
    data_path = os.path.join(cfg_impl.path, processed_dataset_dir)
    if list(cfg_data.sources.values())[0]["provider"] == "fake":
        # Shortcut for fake data
        return _load_fake_dataset(cfg_data, list(cfg_data.sources.values())[0], path=cfg_impl.path)
    elif list(cfg_data.sources.values())[0]["provider"] == "hub":
        return _load_from_hub(cfg_data, data_path)
    else:
        try:
            with main_process_first():
                if cfg_impl.local_staging_dir is not None:
                    data_path = stage_dataset(data_path, cfg_impl.local_staging_dir)
                # Load already processed dataset
                tokenized_dataset = datasets.load_from_disk(data_path)
                tokenizer = load_tokenizer(
                    os.path.join(data_path, "tokenizer"),
                    seq_length=cfg_data.seq_length,
                    vocab_size=cfg_data.vocab_size,
                    cache_dir=cfg_impl.path,
                )
        except FileNotFoundError:
            if cfg_impl.forbid_dataset_preprocessing:
                raise ValueError(
                    f"Cannot find processed at path {data_path}. Dataset preprocessing disabled. "
                    "Dataset preprocessing can be enabled with 'impl.forbid_dataset_preprocessing=False'."
                )
            # Run preprocessing to create dataset
            with main_process_first():
                num_threads = min(torch.get_num_threads(), cfg_impl.threads)  # Mitigate worker overloading
                preprocessed_dataset, new_tokenizer = preprocess_dataset(
                    cfg_data,
                    download_path=cfg_impl.path,
                    num_threads=num_threads,
                    max_raw_chunk_size=cfg_impl.max_raw_chunk_size,
                )

                def save_corpus(path):
                    preprocessed_dataset.save_to_disk(path)
                    new_tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
                    with open(os.path.join(path, "model_config.json"), "w") as file:
                        json.dump(OmegaConf.to_container(cfg_data, resolve=True), file)

                if not cfg_impl.temporary_corpus:
                    # Save to base directory:
                    save_corpus(os.path.join(cfg_impl.path, processed_dataset_dir))
                    if cfg_impl.local_staging_dir is not None:
                        # Optionally also copy into local staging directory
                        data_path = stage_dataset(data_path, cfg_impl.local_staging_dir)
                else:
                    # Directly use staging directory
                    save_corpus(os.path.join(cfg_impl.local_staging_dir, processed_dataset_dir))

            # Reload dataset
            tokenized_dataset = datasets.load_from_disk(data_path)
            tokenizer = load_tokenizer(
                os.path.join(data_path, "tokenizer"),
                seq_length=cfg_data.seq_length,
                vocab_size=cfg_data.vocab_size,
                cache_dir=cfg_impl.path,
            )

    # Cast to tensors after loading from arrow:
    tokenized_dataset.set_format("torch")

    # 4) Log overviews so we always know what's going on with weird tokenization tricks
    random_sentence_idx = torch.randint(0, len(tokenized_dataset), (1,)).item()
    input_data = tokenized_dataset[random_sentence_idx]["input_ids"].squeeze()  # squeeze because hf has leading dim
    dataset_size = len(tokenized_dataset)

    log.info(f"Random sentence with seq_length {tokenizer.model_max_length} from dataset of size {dataset_size:,}: ...")
    log.info(tokenizer.batch_decode(input_data[None])[0])
    log.info("... is tokenized into ...")
    log.info("_".join(tokenizer.decode(t) for t in input_data))
    return tokenized_dataset, tokenizer


def preprocess_dataset(cfg_data, download_path, num_threads=1, max_raw_chunk_size=1e14):
    """A lot of loading and preprocessing."""
    # 1) Collect raw source datasets
    raw_datasets = []
    for name, details in cfg_data.sources.items():
        log.info(f"Now preparing source {name}...")
        if details.provider == "huggingface":
            hf_dataset_settings = {
                k: v for k, v in details.items() if k in ["name", "partition", "split", "language", "date", "beam_runner"] and v is not None
            }
            raw_dataset = datasets.load_dataset(
                name,
                **hf_dataset_settings,
                cache_dir=download_path,
                streaming=details.streaming,
            )
        elif details.provider == "local":
            raw_dataset = datasets.load_dataset(details.file_type, data_files=details.files, streaming=details.streaming)[details.split]
        else:
            raise ValueError(f"Invalid data provider {details.provider} given.")

        # remove columns that break later processing steps
        if details.remove_columns is not None:
            raw_dataset = raw_dataset.remove_columns(details.remove_columns)
        # Filter?
        if getattr(details, "filter", None) is not None:

            def filter_fn(entry):
                """Assume a metadata key 'meta' is present"""
                for key, values in details.filter.items():
                    if entry["meta"][key] in values:
                        return True
                return False

            raw_dataset = raw_dataset.filter(filter_fn)
        # move streams to fixed datasets to make everything sane (and to allow concatenation with unstreamed data)
        if details.streaming:
            raw_dataset = raw_dataset.take(int(cfg_data.max_entries_in_raw_dataset))
            raw_dataset = _move_stream_to_fixed_map(raw_dataset, cfg_data.max_entries_in_raw_dataset, max_raw_chunk_size)
        else:
            if cfg_data.max_entries_in_raw_dataset < len(raw_dataset):
                raw_dataset = raw_dataset.select(range(int(cfg_data.max_entries_in_raw_dataset)))
        # concatenate dataset that were cut into pieces that are too small
        if details.concatenate_successive_entries > 0:
            raw_dataset = _concatenate_entries(raw_dataset, details.concatenate_successive_entries, num_threads=num_threads)
        raw_datasets += [raw_dataset]

    # 2) Preprocess and tokenize
    raw_data = datasets.concatenate_datasets(raw_datasets)
    raw_data = raw_data.shuffle(seed=89)  # Shuffle once here so that multiproc has shards of similar size!
    # This shuffle is crucial for fast multiprocessing tokenization
    # because datasets.map uses a contiguous sharding under the hood.

    # However, we also shuffle so we can now select a smaller range:
    if cfg_data.max_entries_in_raw_dataset < len(raw_data):
        raw_data = raw_data.select(range(int(cfg_data.max_entries_in_raw_dataset)))

    raw_data = raw_dataset_preprocessing(raw_data, num_threads, cfg_data)  # This is by default a no-op, but can be dedup, filtering...
    tokenizer = construct_tokenizer(raw_data, cfg_data, path=download_path)
    tokenized_dataset = _huggingface_preprocessing(raw_data, tokenizer, cfg_data, num_threads=num_threads)  # Tokenize, group, sort...

    return tokenized_dataset, tokenizer


def _move_stream_to_fixed_map(raw_data_streamed, max_entries_in_raw_dataset, max_raw_chunk_size=1e14):
    """Save streaming dataset to a fixed mapping-style database."""
    # I'm tired of IterableDatasets and will take the performance hit to write them out instead:
    try:
        if max_raw_chunk_size > max_entries_in_raw_dataset:
            with tempfile.TemporaryDirectory() as tmpdirname:
                datasets.Dataset.from_dict(dict(text=[v["text"] for v in raw_data_streamed])).save_to_disk(tmpdirname + "raw_data")
                raw_data_mapped = datasets.load_from_disk(tmpdirname + "raw_data")
            # This used to be only a move into RAM but this breaks memory later using C4:
            # raw_data = datasets.Dataset.from_dict(dict(text=[v["text"] for v in raw_data]))
            return raw_data_mapped
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                mapped_sets = []
                data_in_RAM = defaultdict(list)
                for idx, value_stream in enumerate(raw_data_streamed):
                    data_in_RAM["text"].append(value_stream["text"])
                    if ((idx + 1) % max_raw_chunk_size == 0) or ((idx - 1) == max_entries_in_raw_dataset):
                        datasets.Dataset.from_dict(data_in_RAM).save_to_disk(tmpdirname + "raw_data" + str(idx))
                        mapped_dataset = datasets.load_from_disk(tmpdirname + "raw_data" + str(idx))
                        log.info(
                            f"Saved temporary copy at idx {idx} of {max_entries_in_raw_dataset} at {tmpdirname + 'raw_data' + str(idx)}."
                        )
                        data_in_RAM["text"] = []
                        mapped_sets.append(mapped_dataset)
            return datasets.concatenate_datasets(mapped_sets)
    except OSError as e:
        detailed_OSError(e)


def _huggingface_preprocessing(raw_dataset, tokenizer, cfg_data, num_threads=4):
    """Dataset preprocessing and tokenization.

    This is basically the default HF routine from
    https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
    """
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = getattr(raw_dataset, "column_names", "text")
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = tokenizer.model_max_length
    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=num_threads if num_threads > 0 else None,
        # load_from_cache_file=False,
        # keep_in_memory=False,
    )
    parellism_flag = os.environ["TOKENIZERS_PARALLELISM"]
    if num_threads > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # The Collator is modified not to read special_masks anyway:

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            return_special_tokens_mask=False,
            return_attention_mask=False,
            return_token_type_ids=cfg_data.use_type_ids,
        )

    tokenizer.model_max_length = 1e30
    tokenized_dataset = raw_dataset.map(
        tokenize_function, remove_columns=column_names, desc="Running tokenizer on every text in dataset", **map_setup
    )
    tokenizer.model_max_length = max_seq_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)] for k, t in concatenated_examples.items()}
        return result

    tokenized_dataset = tokenized_dataset.map(group_texts, desc=f"Grouping texts in chunks of {max_seq_length}", **map_setup)

    # Shuffle?
    if cfg_data.ordering == "randomized":
        tokenized_dataset = tokenized_dataset.shuffle(seed=233)
    elif cfg_data.ordering == "unigram-curriculum":
        tokenized_dataset = _sort_tokenized_dataset_by_unigram(tokenized_dataset, tokenizer, num_threads)
    elif cfg_data.ordering == "reverse-unigram-curriculum":
        tokenized_dataset = _sort_tokenized_dataset_by_unigram(tokenized_dataset, tokenizer, num_threads, reverse=True)
    elif cfg_data.ordering == "word-length-curriculum":
        tokenized_dataset = _sort_tokenized_dataset_by_word_length(tokenized_dataset, tokenizer, num_threads)
    elif cfg_data.ordering == "sentence-length-curriculum":
        tokenized_dataset = _sort_tokenized_dataset_by_token(tokenized_dataset, tokenizer, tokenizer.encode(".")[0], num_threads)
    elif cfg_data.ordering == "fragment-curriculum":
        tokenized_dataset = _sort_tokenized_dataset_by_token(tokenized_dataset, tokenizer, tokenizer.encode("<sep>")[0], num_threads)
    else:
        raise ValueError(f"Invalid dataset ordering {cfg_data.ordering} provided.")

    # Reduce size to maximal limit:
    if cfg_data.max_seq_in_tokenized_dataset < len(tokenized_dataset):
        tokenized_dataset = tokenized_dataset.select(range(int(cfg_data.max_seq_in_tokenized_dataset)), keep_in_memory=True)

    # Finally flatten
    # This is necessary for the save_to_disk call that comes next. If skipped here, the call will be invoked from save_to_disk
    # This way, atleast it shares the same batch parameters and prints a progress bar.
    tokenized_dataset = tokenized_dataset.map(desc="Flattening the indices", **map_setup)
    os.environ["TOKENIZERS_PARALLELISM"] = parellism_flag
    return tokenized_dataset


def _load_fake_dataset(cfg_data, details, path=None):
    tokenizer = load_tokenizer(cfg_data.tokenizer, cfg_data.seq_length, cfg_data.vocab_size, cache_dir=path)
    tokenizer.model_max_length = cfg_data.seq_length
    generator = torch.Generator()
    generator.manual_seed(details.randgen_seed)
    dataset = torch.randint(0, cfg_data.vocab_size, (details.size, cfg_data.seq_length), generator=generator)
    return dataset, tokenizer


def _concatenate_entries(dataset, num_entries_in_group, num_threads):
    parellism_flag = os.environ["TOKENIZERS_PARALLELISM"]
    if num_threads > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def group_texts(examples):
        result = dict()
        for key, entries in examples.items():
            reduced_list = []
            state, num_collected = None, 0
            for entry in entries:
                num_collected += 1
                if num_collected == 1:
                    state = entry
                else:
                    state += entry
                if num_collected == num_entries_in_group:
                    reduced_list.append(state)
                    state, num_collected = None, 0

            result[key] = reduced_list

        return result

    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=num_threads if num_threads > 0 else None,
        # load_from_cache_file=False,
        # keep_in_memory=True,
    )
    dataset = dataset.map(group_texts, desc="Concatenating examples", **map_setup)
    os.environ["TOKENIZERS_PARALLELISM"] = parellism_flag
    return dataset


# Labels for en_core_web_sm:
SPACY_NER_LABELS = [
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
]


def raw_dataset_preprocessing(raw_dataset, num_threads, cfg_data):
    """Some dataset "improvements". These are optional filtering or normalization rules that are only applied to the pretraining corpus.
    This separates them from generic normalizations that are baked into the tokenizer."""
    column_names = getattr(raw_dataset, "column_names", "text")
    text_column_name = "text" if "text" in column_names else column_names[0]
    known_tokens = []
    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=None,  # a bit messy but c4 in RAM can be overbearing otherwise
        # load_from_cache_file=False,
        # keep_in_memory=False,
    )
    parellism_flag = os.environ["TOKENIZERS_PARALLELISM"]
    if num_threads > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if cfg_data.named_entity_simplification:
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
        nlp.add_pipe("merge_entities")

        def named_entity_simplification(examples):
            # https://stackoverflow.com/questions/58712418/replace-entity-with-its-label-in-spacy
            for idx, doc in enumerate(nlp.pipe(examples[text_column_name], batch_size=1024)):
                examples[text_column_name][idx] = " ".join([t.text if not t.ent_type_ else t.ent_type_ for t in doc])
            return examples

        raw_dataset = raw_dataset.map(named_entity_simplification, desc="Simplify all named entities in dataset.", **map_setup)
        known_tokens += SPACY_NER_LABELS

    if cfg_data.remove_whitespaces:
        # What are you, English-language police?
        def no_whitespaces(examples):
            examples[text_column_name] = ["".join(e.split()) for e in examples[text_column_name]]
            return examples

        raw_dataset = raw_dataset.map(no_whitespaces, desc="Remove any whitespaces.", **map_setup)

    if cfg_data.remove_trash:
        # experimental first test based on Unigram tokenization:
        from transformers import AutoTokenizer

        if cfg_data.remove_trash == "self":
            os.environ["TOKENIZERS_PARALLELISM"] = parellism_flag
            tokenizer = construct_tokenizer(raw_dataset, cfg_data, path=None)
            if num_threads > 0:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
        else:
            tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        tokenizer.model_max_length = 1e30

        def filtering_rule(examples):
            tokenized = tokenizer(examples[text_column_name])["input_ids"]
            return [len(t) < cfg_data.trash_cutoff * len(e) for t, e in zip(tokenized, examples[text_column_name])]

        log.info(f"Size of dataset before trash removal: {len(raw_dataset)}.")
        raw_dataset = raw_dataset.filter(
            filtering_rule,
            desc="Filter sentences that cannot be tokenized well.",
            **map_setup,
            # keep_in_memory=True,  # can run out of mem even on the 750GB node?
        )
        log.info(f"Size of filtered dataset: {len(raw_dataset)}.")

    if cfg_data.deduplicate_entries:
        log.info(f"Size of dataset before deduplication: {len(raw_dataset)}.")
        raw_dataset = deduplicate_huggingface_dataset(
            raw_dataset, threshold=cfg_data.deduplication_threshold, original_cwd=hydra.utils.get_original_cwd()
        )
        log.info(f"Size of deduplicated dataset: {len(raw_dataset)}.")

    os.environ["TOKENIZERS_PARALLELISM"] = parellism_flag
    return raw_dataset


@contextlib.contextmanager
def main_process_first():
    """
    A context manager for torch distributed environment where on needs to do something on the main process, while
    blocking replicas, and when it's finished releasing the replicas.
    One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
    which upon completion saves a cached version of results and which then automatically gets loaded by the
    replicas.

    This is a stripped-down version of the the huggingface context manager from commit 2eb7bb15e771f13192968cd4657c78f76b0799fe
    """
    if torch.distributed.is_initialized():
        is_main_process = torch.distributed.get_rank() == 0
        try:
            if not is_main_process:
                # tell all replicas to wait
                torch.distributed.barrier()
            yield
        finally:
            if is_main_process:
                torch.distributed.barrier()
    else:
        yield


def _load_from_hub(cfg_data, data_path):
    from huggingface_hub import hf_hub_download

    tokenized_dataset = datasets.load_dataset(cfg_data.hf_location, "train", streaming=cfg_data.streaming, cache_dir=data_path)["train"]
    tokenized_dataset = tokenized_dataset.with_format("torch")

    tokenizer_req_files = ["special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]
    os.makedirs(os.path.join(data_path, "tokenizer"), exist_ok=True)
    for file in tokenizer_req_files:
        hf_hub_download(
            cfg_data.hf_location,
            file,
            subfolder="tokenizer",
            repo_type="dataset",
            local_dir=os.path.join(data_path),
        )
    tokenizer = load_tokenizer(os.path.join(data_path, "tokenizer"), cache_dir=data_path)
    return tokenized_dataset, tokenizer
