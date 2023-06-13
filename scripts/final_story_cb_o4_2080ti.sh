
# Final basic BERT runs:
python pretrain.py name=2080tiamp_b512_bert_base_original_final wandb.tags=[bertbase,original,final,bookcorpus] arch=hf-bert-base train=bert-original impl.microbatch_size=32 data=bookcorpus-wikipedia
python eval.py eval=GLUE_sane name=2080tiamp_b512_bert_base_original_final wandb.tags=[bertbase,original,final,bookcorpus] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False

python pretrain.py name=2080tifp32_b512_bert_base_original_final wandb.tags=[bertbase,original,final,bookcorpus] arch=hf-bert-base train=bert-original impl.microbatch_size=32 data=bookcorpus-wikipedia impl.mixed_precision=False
python eval.py eval=GLUE_sane name=2080tifp32_b512_bert_base_original_final wandb.tags=[bertbase,original,final,bookcorpus] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False impl.mixed_precision=False

# Final Izsak comparison:
python pretrain.py name=2080tiamp_b4096_izsak_izsak_final wandb.tags=[bertbase,original,final,bookcorpus] arch=crammed-large-izsak train=bert-izsak impl.microbatch_size=16 data=bookcorpus-wikipedia train.steps=900000
python eval.py eval=GLUE name=2080tiamp_b4096_izsak_izsak_final wandb.tags=[bertbase,original,final,bookcorpus] eval.checkpoint=latest impl.microbatch_size=32 impl.shuffle_in_dataloader=True impl.compile_torch=False


# Final crammed-bert model:
python pretrain.py name=2080tiamp_b8192_cb_o4_final wandb.tags=[o4,final,cb,pile] arch=crammed-bert train=bert-o4 impl.microbatch_size=96 data=the-pile data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False impl._inductor_vars.triton.cudagraphs=False
python eval.py eval=GLUE_sane name=2080tiamp_b8192_cb_o4_final wandb.tags=[pile,o4,cb] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False

# and with script:
python pretrain.py name=2080tiamp_b8192_cb_o4SC_final wandb.tags=[o4,final,cb,pile] arch=crammed-bert train=bert-o4 impl.microbatch_size=32 data=the-pile data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False arch.objective_layout=SCRIPT
python eval.py eval=GLUE_sane name=2080tiamp_b8192_cb_o4SC_final wandb.tags=[pile,o4,cb] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False

# and default inductor config
python pretrain.py name=2080tiamp_b8192_cb_o4ID_final wandb.tags=[o4,final,cb,pile] arch=crammed-bert train=bert-o4 impl.microbatch_size=96 data=the-pile data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False impl._inductor_vars=null
python eval.py eval=GLUE_sane name=2080tiamp_b8192_cb_o4ID_final wandb.tags=[pile,o4,cb] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False


###### Final ablation table:
# no changes to data
python pretrain.py name=2080tiamp_b8192_cb_o4_bookcorpus_final wandb.tags=[o4,final,cb,pile] arch=crammed-bert train=bert-o4 impl.microbatch_size=96 data=bookcorpus-wikipedia  impl._inductor_vars.triton.cudagraphs=False
python eval.py eval=GLUE_sane name=2080tiamp_b8192_cb_o4_bookcorpus_final wandb.tags=[pile,o4,cb] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False

# no changes to training (aside from scheduler running to zero)
python pretrain.py name=2080tiamp_b512_cb_orig_final wandb.tags=[o4,final,cb,pile] arch=crammed-bert train=bert-original train.scheduler=budget-cosine-decay train.warmup_steps=0.06 impl.microbatch_size=96 data=the-pile data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False  impl._inductor_vars.triton.cudagraphs=False
python eval.py eval=GLUE_sane name=2080tiamp_b512_cb_orig_final wandb.tags=[pile,o4,cb] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False

# minimal changes to training setup
python pretrain.py name=2080tiamp_b512_cb_base_final wandb.tags=[o4,final,cb,pile] arch=crammed-bert train=bert-base train.scheduler=budget-cosine-decay train.warmup_steps=0.06 impl.microbatch_size=96 data=the-pile data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False  impl._inductor_vars.triton.cudagraphs=False
python eval.py eval=GLUE_sane name=2080tiamp_b512_cb_base_final wandb.tags=[pile,o4,cb] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False

python pretrain.py name=2080tiamp_b8192_cb_base_final wandb.tags=[o4,final,cb,pile] arch=crammed-bert train.batch_size=8192 train=bert-base train.scheduler=budget-cosine-decay train.warmup_steps=0.06 impl.microbatch_size=96 data=the-pile data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False  impl._inductor_vars.triton.cudagraphs=False
python eval.py eval=GLUE_sane name=2080tiamp_b8192_cb_base_final wandb.tags=[pile,o4,cb] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False

# basic architecture
python pretrain.py name=2080tiamp_b8192_base_o4_final wandb.tags=[o4,final,cb,pile] arch=hf-bert-base train=bert-o4 impl.microbatch_size=32 data=the-pile data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False
python eval.py eval=GLUE_sane name=2080tiamp_b8192_base_o4_final wandb.tags=[pile,o4,cb] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False

# architecture with minimal changes
python pretrain.py name=2080tiamp_b8192_c1_o4_final wandb.tags=[o4,final,cb,pile] arch=crammed-bert-simple train=bert-o4 impl.microbatch_size=64 data=the-pile data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False
python eval.py eval=GLUE_sane name=2080tiamp_b8192_c1_o4_final wandb.tags=[pile,o4,cb] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False


# basic architecture + simple training routine
python pretrain.py name=2080tiamp_b512_base_base_final wandb.tags=[o4,final,cb,pile] arch=hf-bert-base train=bert-original train.scheduler=budget-cosine-decay train.warmup_steps=0.06 impl.microbatch_size=32 data=the-pile data.remove_trash=True data.ordering=sentence-length-curriculum data.deduplicate_entries=False
python eval.py eval=GLUE_sane name=2080tiamp_b512_base_base_final wandb.tags=[pile,o4,cb] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False
