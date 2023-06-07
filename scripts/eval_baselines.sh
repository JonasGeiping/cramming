
# baseline:
python eval.py eval=GLUE name=hf-bert-pretrained base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://bert-base-uncased impl.microbatch_size=32 wandb.tags=[GLUE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False
python eval.py eval=GLUE name=hf-roberta-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=32 wandb.tags=[GLUE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False
python eval.py eval=GLUE name=hf-roberta-large-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=8 wandb.tags=[GLUE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False

python eval.py eval=GLUE_sane name=hf-bert-pretrained base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://bert-base-uncased impl.microbatch_size=16 wandb.tags=[GLUE,baseline,lrablation] impl.shuffle_in_dataloader=True impl.compile_torch=False
python eval.py eval=GLUE_sane name=hf-roberta-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=16 wandb.tags=[GLUE,baseline,lrablation] impl.shuffle_in_dataloader=True impl.compile_torch=False
python eval.py eval=GLUE_sane name=hf-roberta-large-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=8 wandb.tags=[GLUE,baseline,lrablation] impl.shuffle_in_dataloader=True impl.compile_torch=False

# RACE
python eval.py eval=RACE name=hf-bert-pretrained base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://bert-base-uncased impl.microbatch_size=16 wandb.tags=[RACE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512
python eval.py eval=RACE name=hf-roberta-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=16 wandb.tags=[RACE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512
python eval.py eval=RACE name=hf-roberta-large-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=8 wandb.tags=[RACE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512

# superGLUE (mostly)
python eval.py eval=superGLUE name=hf-bert-pretrained base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://bert-base-uncased impl.microbatch_size=16 wandb.tags=[superGLUE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512
python eval.py eval=superGLUE name=hf-roberta-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=16 wandb.tags=[superGLUE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512
python eval.py eval=superGLUE name=hf-roberta-large-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=8 wandb.tags=[superGLUE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512

python eval.py eval=superGLUE eval/tasks=record name=hf-bert-pretrained base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://bert-base-uncased impl.microbatch_size=16 wandb.tags=[superGLUE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512
python eval.py eval=superGLUE eval/tasks=record name=hf-roberta-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=16 wandb.tags=[superGLUE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512
python eval.py eval=superGLUE eval/tasks=record name=hf-roberta-large-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=8 wandb.tags=[superGLUE,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512

# SWAG
python eval.py eval=SWAG name=hf-bert-pretrained base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://bert-base-uncased impl.microbatch_size=16 wandb.tags=[SWAG,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512
python eval.py eval=SWAG name=hf-roberta-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=16 wandb.tags=[SWAG,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512
python eval.py eval=SWAG name=hf-roberta-large-pretrained-cased base_dir=/fs/cml-projects/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=8 wandb.tags=[SWAG,baseline] impl.shuffle_in_dataloader=True impl.compile_torch=False eval.max_seq_length=512 impl.pad_to_multiple_of=512

# Run the currently strongest model on RACE/SWAG/superGLUE, with default parameters
python eval.py eval=SWAG name=A6000amp_b8192_cb_o4_final base_dir=/fs/cml-projects/cramming/ wandb.tags=[bw,o3,c5] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False eval.max_seq_length=128 impl.pad_to_multiple_of=128
python eval.py eval=RACE name=A6000amp_b8192_cb_o4_final base_dir=/fs/cml-projects/cramming/ wandb.tags=[bw,o3,c5] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False eval.max_seq_length=128 impl.pad_to_multiple_of=128
python eval.py eval=superGLUE name=A6000amp_b8192_cb_o4_final base_dir=/fs/cml-projects/cramming/ wandb.tags=[bw,o3,c5] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.compile_torch=False eval.max_seq_length=128 impl.pad_to_multiple_of=128
