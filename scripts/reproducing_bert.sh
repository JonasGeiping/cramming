
# Reproduce BERT with maximal budget:
torchrun --nproc_per_node=8 --standalone pretrain.py name=fp32_b512_bert_repro wandb.tags=[bert_repro] data=bookcorpus-wikipedia arch=bert-original train=bert-original train.batch_size=64 impl.microbatch_size=64 budget=47 train.steps=1000000 train.warmup=10000
torchrun --nproc_per_node=8 --standalone pretrain.py name=fp32_b4096_bert_repro data=bookcorpus-wikipedia arch=bert-original train=bert-original train.batch_size=512 impl.microbatch_size=128 budget=47 train.steps=1000000 train.warmup=10000

torchrun --nproc_per_node=8 --standalone pretrain.py name=amp_b512_bert_repro wandb.tags=[bert_repro] data=bookcorpus-wikipedia arch=bert-original train=bert-original train.batch_size=64 impl.microbatch_size=64 budget=47 train.steps=1000000 train.warmup=10000
torchrun --nproc_per_node=8 --standalone pretrain.py name=amp_b4096_bert_repro wandb.tags=[bert_repro] data=bookcorpus-wikipedia arch=bert-original train=bert-original train.batch_size=512 impl.microbatch_size=128 budget=47 train.steps=1000000 train.warmup=10000

# Corresponding runs with modified archs:
torchrun --nproc_per_node=8 --standalone pretrain.py name=amp_b4096_c0_bw wandb.tags=[bert_repro,c0] data=bookcorpus-wikipedia arch=bert-base train=bert-o2 train.batch_size=512 impl.microbatch_size=256 budget=47 train.steps=800000
torchrun --nproc_per_node=8 --standalone pretrain.py name=amp_b4096_c2_bw wandb.tags=[bert_repro,c2] data=bookcorpus-wikipedia arch=bert-c2 train=bert-o2 train.batch_size=512 impl.microbatch_size=256 budget=47 train.steps=800000
torchrun --nproc_per_node=8 --standalone pretrain.py name=amp_b4096_c3_bw wandb.tags=[bert_repro,c3] data=bookcorpus-wikipedia arch=bert-c3 train=bert-o2 train.batch_size=512 impl.microbatch_size=256 budget=47 train.steps=800000
torchrun --nproc_per_node=8 --standalone pretrain.py name=amp_b4096_c4_bw wandb.tags=[bert_repro,c4] data=bookcorpus-wikipedia arch=bert-c4 train=bert-o2 train.batch_size=512 impl.microbatch_size=256 budget=47 train.steps=800000
torchrun --nproc_per_node=8 --standalone pretrain.py name=amp_b4096_c5_bw wandb.tags=[bert_repro,c5] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=512 impl.microbatch_size=256 budget=47 train.steps=800000

# Iszak:
torchrun --nproc_per_node=8 --standalone pretrain.py name=amp_b4096_izsak wandb.tags=[bert_repro,izsak] data=bookcorpus-wikipedia arch=bert-large-izsak train=bert-izsak train.batch_size=512 impl.microbatch_size=128 budget=47 train.steps=800000
