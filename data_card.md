---
dataset_info:
  features:
  - name: input_ids
    sequence: int32
  splits:
  - name: train
    num_bytes: 22274051772
    num_examples: 43166767
  download_size: 12187746609
  dataset_size: 22274051772
  annotations_creators:
  - no-annotation
  language_creators:
  - found
  language:
  - en
  license: other
  multilinguality:
  - monolingual
  pretty_name: pretokenized,filtered,sorted subset of the Pile
  size_categories:
  - 10B<n<100B
  source_datasets:
  - the-pile
  task_categories:
  - text-generation
  - fill-mask
  task_ids:
  - language-modeling
  - masked-language-modeling
  paperswithcode_id: the-pile-cramming

---
# Dataset Card for "the_pile_WordPiecex32768_97b8e776baafb99c3892e6572a9f51b3"


## Dataset Description

- **Repository:** https://github.com/JonasGeiping/cramming
- **Paper:** https://arxiv.org/abs/2212.14034
- **Raw Data Source Paper:** [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)
- **Raw Data Source Datasheet:** [Datasheet for the Pile](https://arxiv.org/abs/2201.07311)

### Dataset Summary

This is a preprocessed, tokenized dataset for the cramming-project.

Use only with the tokenizer uploaded here.
This version is `97b8e776baafb99c3892e6572a9f51b3`, which corresponds to a specific dataset construction setup, described below.
The raw data source is the Pile, a 825 GiB diverse, open source language modelling data set that consists of 22 smaller, high-quality
datasets combined together.


### Languages

This dataset is in English (`EN`).

### Data Splits

This preprocessed subset contains only a train split.

## Dataset Creation

The configuration to create this dataset with the cramming project code (https://github.com/JonasGeiping/cramming) is

```
# This is a slice of the pile, loaded from a local source
name: the_pile
defaults:
  - sources:
      - the_pile

#
# Preprocessing
normalizer:
  force_lowercase: True
  strip_accents: True
  force_english_keyboard: True
  whitespace_escape: False
tokenizer: WordPiece
vocab_size: 32768

# Dataset Formation
seq_length: 128
include_cls_token_in_corpus: False
include_sep_token_in_corpus: True
use_type_ids: False
max_entries_in_raw_dataset: 16e6 # About 40 mio seqs of length 128
max_seq_in_tokenized_dataset: 85e6 # Select only this many tokenized sequences.
# max_seq_in_tokenized_dataset should be just slightly more than budget * 60 * 60 * expected tokens/sec for the single epoch of training

# Data Cleaning:
named_entity_simplification: False
remove_whitespaces: False
remove_trash: True
trash_cutoff: 0.25
deduplicate_entries: False
deduplication_threshold: 75

# Data Order:
ordering: sentence-length-curriculum # could be a curriculum

```

## Considerations for Using the Data

Limitations and bias:
This training data was further filtered and sorted beyond the normal preprocessing.
These modifications were not tested for unintended consequences.

## Additional Information

### Dataset Curators

This dataset is a filtered, sorted and preprocessed subset of the the-Pile made by Jonas Geiping . The original dataset was primarily curated by Leo Gao and Stella Biderman, with assistance from other authors of the Pile paper.

### Licensing Information

Please refer to the specific license depending on the subset you use at https://huggingface.co/datasets/EleutherAI/pile

### Citation Information

```
@article{gao2020pile,
  title={The {P}ile: An 800{GB} dataset of diverse text for language modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and others},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
@article{biderman2022datasheet,
  title={Datasheet for the pile},
  author={Biderman, Stella and Bicheno, Kieran and Gao, Leo},
  journal={arXiv preprint arXiv:2201.07311},
  year={2022}
}
