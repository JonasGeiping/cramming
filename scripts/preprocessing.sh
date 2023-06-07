# CPU jobs to preprocess data
# 4xVariations for all datasets
# You will not need all of these, this is just for reference.
# Probably about 2-2.5TB to store all of these variations.

# all raw:
python pretrain.py name=bw_raw data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=False data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=c4_raw data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=False data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=oscar_raw data=oscar dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=False data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=roots_raw data=roots-mini dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=False data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=owt_raw data=openweb dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=False data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=pile1_raw data=the-pile dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=False data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=pile2_raw data=the-pile-natural dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=False data.ordering=randomized data.deduplicate_entries=False

# # all filtered (with albert for speed/simplicity)
python pretrain.py name=bw_filt data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=c4_filt data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=oscar_filt data=oscar dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=roots_filt data=roots-mini dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=owt_filt data=openweb dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=pile1_filt data=the-pile dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=randomized data.deduplicate_entries=False
python pretrain.py name=pile2_filt data=the-pile-natural dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=randomized data.deduplicate_entries=False

# # and sorted
python pretrain.py name=bw_filtsort data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False
python pretrain.py name=c4_filtsort data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False
python pretrain.py name=oscar_filtsort data=oscar dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False
python pretrain.py name=roots_filtsort data=roots-mini dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False
python pretrain.py name=owt_filtsort data=openweb dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False
python pretrain.py name=pile1_filtsort data=the-pile dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False
python pretrain.py name=pile2_filtsort data=the-pile-natural dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False

# # and deduped
python pretrain.py name=bw_filtsortdedup data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=True
python pretrain.py name=c4_filtsortdedup data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=True
python pretrain.py name=oscar_filtsortdedup data=oscar dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=True
python pretrain.py name=roots_filtsortdedup data=roots-mini dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=True
python pretrain.py name=owt_filtsortdedup data=openweb dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=True
python pretrain.py name=pile1_filtsortdedup data=the-pile dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=True
python pretrain.py name=pile2_filtsortdedup data=the-pile-natural dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=True


# Differing vocab sizes:
python pretrain.py name=bookcorpus_wiki data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=2048
python pretrain.py name=bookcorpus_wiki data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=4096
python pretrain.py name=bookcorpus_wiki data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=8192
python pretrain.py name=bookcorpus_wiki data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=16384
python pretrain.py name=bookcorpus_wiki data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=32768
python pretrain.py name=bookcorpus_wiki data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=65536
python pretrain.py name=bookcorpus_wiki data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=131072

# Another variant with double the vocab size for C4
python pretrain.py name=c4_trash025_dedup75_sent_sorted65k data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6 data.vocab_size=65536


# C4-specific variations (branching off from filtered+sorted):
python pretrain.py name=c4filtsort_base data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False
python pretrain.py name=c4filtsort_selftrash data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False data.remove_trash=self
python pretrain.py name=c4filtsort_V65 data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False data.vocab_size=65536
python pretrain.py name=c4filtsort_V131 data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False data.vocab_size=131072
python pretrain.py name=c4filtsort_CLS data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False data.include_cls_token_in_corpus=True
python pretrain.py name=c4filtsort_bertok data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False data.tokenizer=bert-base-uncased data.vocab_size=30522
python pretrain.py name=c4filtsort_sbpe data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False data.tokenizer=SentencePieceBPE

python pretrain.py name=c4filtsort_seq512 data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False data.seq_length=512
python pretrain.py name=c4filtsort_t02 data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False data.trash_cutoff=0.2 data.max_entries_in_raw_dataset=40e6
python pretrain.py name=c4filtsort_fragsort data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False data.ordering=fragment-curriculum
python pretrain.py name=c4filtsort_probsort data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False data.ordering=unigram-curriculum
python pretrain.py name=c4filtsort_dedup50 data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=True data.deduplication_threshold=50
python pretrain.py name=c4filtsort_dedup100 data=c4-subset-processed dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=False data.deduplication_threshold=100
