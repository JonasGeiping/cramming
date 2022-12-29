
# crammed BERT
python eval.py eval=GLUE_sane name=amp_b4096_c5_o3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[cc,a6000] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5



# # baseline:
python eval.py eval=GLUE name=hf-bert-pretrained base_dir=/cmlscratch/jonas0/cramming/ eval.checkpoint=hf://bert-base-uncased impl.microbatch_size=32 wandb.tags=[GLUE,baseline] impl.shuffle_in_dataloader=True
python eval.py eval=GLUE name=hf-roberta-pretrained-cased base_dir=/cmlscratch/jonas0/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=32 wandb.tags=[GLUE,baseline] impl.shuffle_in_dataloader=True
python eval.py eval=GLUE name=hf-roberta-large-pretrained-cased base_dir=/cmlscratch/jonas0/cramming/ eval.checkpoint=hf://roberta-base impl.microbatch_size=8 wandb.tags=[GLUE,baseline] impl.shuffle_in_dataloader=True

python eval.py eval=GLUE name=hf-bert-random base_dir=/cmlscratch/jonas0/cramming/ eval.checkpoint=hf://bert-base-uncased-untrained impl.microbatch_size=32 wandb.tags=[GLUE,baseline] impl.shuffle_in_dataloader=True
python eval.py eval=GLUE name=hf-roberta-random base_dir=/cmlscratch/jonas0/cramming/ eval.checkpoint=hf://roberta-base-untrained impl.microbatch_size=32 wandb.tags=[GLUE,baseline] impl.shuffle_in_dataloader=True

# # # izsak comparison:
# python eval.py eval=GLUE_sane name=amp_b4096_izsak_izsak base_dir=/cmlscratch/jonas0/cramming/ eval.checkpoint=latest impl.shuffle_in_dataloader=True wandb.tags=[GLUE,izsak] impl.microbatch_size=4
# python eval.py eval=GLUE_sane name=amp_b4096_original_izsak base_dir=/cmlscratch/jonas0/cramming/ eval.checkpoint=latest impl.shuffle_in_dataloader=True wandb.tags=[GLUE,izsak] impl.microbatch_size=4
