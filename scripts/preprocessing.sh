# CPU jobs to preprocess data

# baselines:
python pretrain.py name=bookcorpus_wiki base_dir=/cmlscratch/jonas0/cramming/ data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.max_seq_in_tokenized_dataset=85e6
python pretrain.py name=thepile1 base_dir=/cmlscratch/jonas0/cramming/ data=the-pile dryrun=True impl.forbid_dataset_preprocessing=False
python pretrain.py name=thepile2 base_dir=/cmlscratch/jonas0/cramming/ data=the-pile-natural dryrun=True impl.forbid_dataset_preprocessing=False
python pretrain.py name=c4 base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.max_seq_in_tokenized_dataset=85e6
#
#
# # with filtering:
python pretrain.py name=bookcorpus_wiki_trash025 base_dir=/cmlscratch/jonas0/cramming/ data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25
python pretrain.py name=thepile1_trash025 base_dir=/cmlscratch/jonas0/cramming/ data=the-pile dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25
python pretrain.py name=thepile2_trash025 base_dir=/cmlscratch/jonas0/cramming/ data=the-pile-natural dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25
python pretrain.py name=c4_trash025 base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25

# with deduplication:
python pretrain.py name=bookcorpus_wiki_dedup75 base_dir=/cmlscratch/jonas0/cramming/ data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=True data.deduplication_threshold=75
python pretrain.py name=thepile1_dedup50 base_dir=/cmlscratch/jonas0/cramming/ data=the-pile dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=True data.deduplication_threshold=75
python pretrain.py name=thepile2_dedup75 base_dir=/cmlscratch/jonas0/cramming/ data=the-pile-natural dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=True data.deduplication_threshold=75
python pretrain.py name=c4_dedup50 base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.deduplicate_entries=True data.deduplication_threshold=75

# filtering and sorting
python pretrain.py name=c4_trash025_uni_sorted base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25 data.ordering=unigram-curriculum
python pretrain.py name=c4_trash025_word_sorted base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25 data.ordering=word-length-curriculum
python pretrain.py name=c4_trash025_sent_sorted base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum
python pretrain.py name=c4_trash025_frag_sorted base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25 data.ordering=fragment-curriculum

# Differing vocab sizes:
python pretrain.py name=bookcorpus_wiki base_dir=/cmlscratch/jonas0/cramming/ data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=2048
python pretrain.py name=bookcorpus_wiki base_dir=/cmlscratch/jonas0/cramming/ data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=4096
python pretrain.py name=bookcorpus_wiki base_dir=/cmlscratch/jonas0/cramming/ data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=8192
python pretrain.py name=bookcorpus_wiki base_dir=/cmlscratch/jonas0/cramming/ data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=16384
python pretrain.py name=bookcorpus_wiki base_dir=/cmlscratch/jonas0/cramming/ data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=32768
python pretrain.py name=bookcorpus_wiki base_dir=/cmlscratch/jonas0/cramming/ data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=65536
python pretrain.py name=bookcorpus_wiki base_dir=/cmlscratch/jonas0/cramming/ data=bookcorpus-wikipedia dryrun=True impl.forbid_dataset_preprocessing=False data.vocab_size=131072

# filtering and sorting and deduplication of c4 (large dataset for the a6000 cards)
python pretrain.py name=c4_trash025_dedup75_sent_sorted base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python pretrain.py name=c4_dedup75_sent_sorted base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=False data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python pretrain.py name=c4_trash025_dedup75 base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25 data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python pretrain.py name=c4_trash025_sent_sorted base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.max_seq_in_tokenized_dataset=85e6
python pretrain.py name=c4_sent_sorted base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=False data.ordering=sentence-length-curriculum data.max_seq_in_tokenized_dataset=85e6

# Do one variant with double the vocab size for C4
python pretrain.py name=c4_trash025_dedup75_sent_sorted65k base_dir=/cmlscratch/jonas0/cramming/ data=c4-subset dryrun=True impl.forbid_dataset_preprocessing=False data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6 data.vocab_size=65536
