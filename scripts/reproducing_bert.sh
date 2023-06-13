
# Reproduce BERT with maximal budget:
torchrun --nproc_per_node=8 --standalone pretrain.py name=fp32_b512_bert_repro wandb.tags=[bert_repro] data=bookcorpus-wikipedia arch=bert-original train=bert-original train.batch_size=64 impl.microbatch_size=64 budget=19200 train.steps=1000000 train.warmup=10000
torchrun --nproc_per_node=8 --standalone pretrain.py name=fp32_b4096_bert_repro data=bookcorpus-wikipedia arch=bert-original train=bert-original train.batch_size=512 impl.microbatch_size=128 budget=47 train.steps=1000000 train.warmup=10000

torchrun --nproc_per_node=8 --standalone pretrain.py name=amp_b512_bert_repro wandb.tags=[bert_repro] data=bookcorpus-wikipedia arch=bert-original train=bert-original train.batch_size=64 impl.microbatch_size=64 budget=47 train.steps=1000000 train.warmup=10000
torchrun --nproc_per_node=8 --standalone pretrain.py name=amp_b4096_bert_repro wandb.tags=[bert_repro] data=bookcorpus-wikipedia arch=bert-original train=bert-original train.batch_size=512 impl.microbatch_size=128 budget=19200 train.steps=1000000 train.warmup=10000
