# # original
python pretrain.py name=A6000amp_b4096_orig_orig_final wandb.tags=[bookcorpus] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 data=bookcorpus-wikipedia arch=bert-original train=bert-original impl.microbatch_size=64 train.batch_size=4096
python eval.py eval=GLUE name=A6000amp_b4096_orig_orig_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus] eval.checkpoint=latest impl.microbatch_size=32

# izsak
python pretrain.py name=A6000amp_b4096_izsak_izsak_final wandb.tags=[bookcorpus] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 data=bookcorpus-wikipedia arch=bert-large-izsak train=bert-izsak impl.microbatch_size=64 train.batch_size=4096
python eval.py eval=GLUE name=A6000amp_b4096_izsak_izsak_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus] eval.checkpoint=latest impl.microbatch_size=32

# final
python pretrain.py name=A6000amp_b4096_c5_o3_final wandb.tags=[o3,cc,c5] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A6000amp_b4096_c5_o3_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[cc,o3,c5] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# no changes to data
python pretrain.py name=A6000amp_b4096_c5_o3_bookcorpus_final wandb.tags=[bookcorpus,o3,c5] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 arch=bert-c5 train=bert-o3 train.batch_size=4096 data=bookcorpus-wikipedia
python eval.py eval=GLUE_sane name=A6000amp_b4096_c5_o3_bookcorpus_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,o3,c5] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# no changes to training (aside from scheduler running to zero)
python pretrain.py name=A6000amp_b4096_c5_orig_final wandb.tags=[cc,c5] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 arch=bert-c5 train=bert-original train.scheduler=budget-cosine-decay train.warmup_steps=0.06 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A6000amp_b4096_c5_orig_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[cc,c5] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# basic training setup
python pretrain.py name=A6000amp_b4096_c5_o0_final wandb.tags=[cc,c5] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 arch=bert-c5 train=bert-base train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A6000amp_b4096_c5_o0_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[cc,c5] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# # basic architecture
# # crashes occasionally with nonfinite loss
python pretrain.py name=A6000amp_b4096_orig_o3_final wandb.tags=[o3,cc] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 arch=bert-original train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A6000amp_b4096_orig_o3_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[cc,o3] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5


# # basic architecture + simple training routine
python pretrain.py name=A6000amp_b4096_orig_orig_final wandb.tags=[o3,cc] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 arch=bert-original train=bert-original train.scheduler=budget-cosine-decay train.warmup_steps=0.06 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A6000amp_b4096_orig_orig_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[cc,o3] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# architecture with minimal changes
python pretrain.py name=A6000amp_b4096_c0pre_o3_final wandb.tags=[o3,cc] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 arch=bert-base arch.norm_scheme=pre train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6
python eval.py eval=GLUE_sane name=A6000amp_b4096_c0pre_o3_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[cc,o3] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# basic architecture + flash
python pretrain.py name=A6000amp_b4096_origflash_o3_final wandb.tags=[o3,cc] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 arch=bert-original train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6 arch.attention.type=flash-attention-impl arch.attention.high_level_fusion=False
python eval.py eval=GLUE_sane name=A6000amp_b4096_origflash_o3_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[cc,o3] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# architecture with minimal changes + flash
python pretrain.py name=A6000amp_b4096_c0flashpre_o3_final wandb.tags=[o3,cc] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 arch=bert-base arch.norm_scheme=pre train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6 arch.attention.type=flash-attention-impl arch.attention.high_level_fusion=False
python eval.py eval=GLUE_sane name=A6000amp_b4096_c0flashpre_o3_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[cc,o3] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# ablate to bamp
python pretrain.py name=A6000bamp_b4096_c5_o3_final wandb.tags=[o3,cc,c5] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch1/jonas0 arch=bert-c5 train=bert-o3 train.batch_size=4096 data=c4-subset data.remove_trash=True data.trash_cutoff=0.25 data.ordering=sentence-length-curriculum data.deduplicate_entries=True data.deduplication_threshold=75 data.max_seq_in_tokenized_dataset=85e6 impl.mixed_precision_target_dtype=bfloat16
python eval.py eval=GLUE_sane name=A6000bamp_b4096_c5_o3_final base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[cc,o3,c5] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.mixed_precision_target_dtype=bfloat16
