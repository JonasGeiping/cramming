#
# These are instructions for the old, pytorch 1.13, version of the repo, they are not usable on the updated checkpoint and provided only for reference.
# (but could be easily adapted to the newer format)
#


#
# final
python pretrain.py name=A4000amp_b4096_c5_o3_data_trash025_dedup75_sent_sorted wandb.tags=[o3,cc,c5,data] arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A4000amp_b4096_c5_o3_data_trash025_dedup75_sent_sorted wandb.tags=[cc,o3,c5,data] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=A4000amp_b4096_c5_o3_data_bookcorpus_wiki wandb.tags=[o3,cc,c5,data] arch=bert-c5 train=bert-o3 train.batch_size=4096 data=bookcorpus-wikipedia
python eval.py eval=GLUE_sane name=A4000amp_b4096_c5_o3_bookcorpus_wiki wandb.tags=[cc,o3,c5,data] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=A4000amp_b4096_c5_o3_data_c4 wandb.tags=[o3,cc,c5,data] arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A4000amp_b4096_c5_o3_data_c4 wandb.tags=[cc,o3,c5,data] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=A4000amp_b4096_c5_o3_data_c4_dedup75_sent_sorted wandb.tags=[o3,cc,c5,data] arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A4000amp_b4096_c5_o3_data_c4_dedup75_sent_sorted wandb.tags=[cc,o3,c5,data] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=A4000amp_b4096_c5_o3_data_c4_trash025_dedup75 wandb.tags=[o3,cc,c5,data] arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A4000amp_b4096_c5_o3_data_c4_trash025_dedup75 wandb.tags=[cc,o3,c5,data] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=A4000amp_b4096_c5_o3_data_c4_trash025_sent_sorted wandb.tags=[o3,cc,c5,data] arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A4000amp_b4096_c5_o3_data_c4_trash025_sent_sorted wandb.tags=[cc,o3,c5,data] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=A4000amp_b4096_c5_o3_data_c4_sent_sorted wandb.tags=[o3,cc,c5,data] arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=False data.ordering=sentence-length-curriculum data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A4000amp_b4096_c5_o3_data_c4_sent_sorted wandb.tags=[cc,o3,c5,data] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=A4000amp_b4096_c5_o3_data_c4_trash025_dedup75_sent_sorted65k wandb.tags=[o3,cc,c5,data] arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6 data.vocab_size=65536
python eval.py eval=GLUE_sane name=A4000amp_b4096_c5_o3_data_c4_trash025_dedup75_sent_sorted65k wandb.tags=[cc,o3,c5,data] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=A4000amp_b4096_c5_o3_data_c4_trash025_dedup75_sent_sorted65k wandb.tags=[o3,cc,c5,data] arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6 data.vocab_size=65536
python eval.py eval=GLUE_sane name=A4000amp_b4096_c5_o3_data_c4_trash025_dedup75_sent_sorted65k wandb.tags=[cc,o3,c5,data] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=A4000amp_b4096_c5_o3_data_c4_trash025_dedup75_sent_sorted65k wandb.tags=[o3,cc,c5,data] arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6 data.vocab_size=65536
python eval.py eval=GLUE_sane name=A4000amp_b4096_c5_o3_data_c4_trash025_dedup75_sent_sorted65k wandb.tags=[cc,o3,c5,data] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5
