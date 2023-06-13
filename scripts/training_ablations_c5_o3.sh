#
# These are instructions for the old, pytorch 1.13, version of the repo, they are not usable on the updated repo and provided only for reference.
# (but could be easily adapted to the newer format)
#

# # Run through the entire training setup and reproduce numbers on c5
#
# coarse:
python pretrain.py name=amp_b4096_orig_orig wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-original train=bert-original impl.microbatch_size=64 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_orig_orig wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_orig wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-original train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_orig wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-base train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o1 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o1 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o1 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o2 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o2 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o2 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_izsak wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-izsak train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_izsak wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5


#### sequence curriculum
python pretrain.py name=amp_b4096_c5_o3_CU1 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 +train.sequence_curriculum.lengths=[8,16,32,64,128] +train.sequence_curriculum.triggers=[0.1,0.2,0.3,0.5,0.75] +train.sequence_curriculum.unfold=False
python eval.py eval=mnli name=amp_b4096_c5_o3_CU1 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_CU2 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 +train.sequence_curriculum.lengths=[8,16,32,64,128] +train.sequence_curriculum.triggers=[0.1,0.2,0.3,0.5,0.75] +train.sequence_curriculum.unfold=True
python eval.py eval=mnli name=amp_b4096_c5_o3_CU2 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_CU3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 +train.sequence_curriculum.lengths=[8,16,32,64,128] +train.sequence_curriculum.triggers=[0.2,0.35,0.5,0.65,0.85] +train.sequence_curriculum.unfold=True
python eval.py eval=mnli name=amp_b4096_c5_o3_CU3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# EMA
python pretrain.py name=amp_b4096_c5_o3_ema0995 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 +train.weight_averaging.frequency=1 +train.weight_averaging.type=EMA +train.weight_averaging.momentum=0.995
python eval.py eval=mnli name=amp_b4096_c5_o3_ema0995 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_ema099 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 +train.weight_averaging.frequency=1 +train.weight_averaging.type=EMA +train.weight_averaging.momentum=0.99
python eval.py eval=mnli name=amp_b4096_c5_o3_ema099 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_ema0999 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 +train.weight_averaging.frequency=1 +train.weight_averaging.type=EMA +train.weight_averaging.momentum=0.999
python eval.py eval=mnli name=amp_b4096_c5_o3_ema0999 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# LAWA
python pretrain.py name=amp_b4096_c5_o3_lawa5000_10 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 +train.weight_averaging.frequency=5000 +train.weight_averaging.type=LAWA +train.weight_averaging.last_k=10
python eval.py eval=mnli name=amp_b4096_c5_o3_lawa5000_10 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_lawa25000_10 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 +train.weight_averaging.frequency=25000 +train.weight_averaging.type=LAWA +train.weight_averaging.last_k=10
python eval.py eval=mnli name=amp_b4096_c5_o3_lawa25000_10 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_lawa1000_10 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 +train.weight_averaging.frequency=1000 +train.weight_averaging.type=LAWA +train.weight_averaging.last_k=10
python eval.py eval=mnli name=amp_b4096_c5_o3_lawa1000_10 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

##### with dropout
python pretrain.py name=amp_b4096_c5_o3_trainmode wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.pretrain_in_train_mode=True
python eval.py eval=mnli name=amp_b4096_c5_o3_trainmode wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

##### MLM ######
python pretrain.py name=amp_b4096_c5_o3_mlm20 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.pretrain_in_train_mode=True train.objective.mlm_probability=0.2
python eval.py eval=mnli name=amp_b4096_c5_o3_mlm20 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_mlm40 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.pretrain_in_train_mode=True train.objective.mlm_probability=0.4
python eval.py eval=mnli name=amp_b4096_c5_o3_mlm40 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4032_c5_o3_mlm60 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4032 train.pretrain_in_train_mode=True train.objective.mlm_probability=0.6 impl.microbatch_size=96
python eval.py eval=mnli name=amp_b4032_c5_o3_mlm60 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5
#
# ####### Batch Sizes
# # fixed BS
python pretrain.py name=amp_b128_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=128 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b128_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b256_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=256 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b256_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b384_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=384 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b384_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b768_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=768 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b768_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b1536_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=1536 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b1536_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b3072_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=3072 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b3072_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b6144_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=6144 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b6144_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b12288_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=12288 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b12288_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# fixed bs300
python pretrain.py name=amp_b96_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=96 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b96_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b192_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=192 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b192_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b384_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=384 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b384_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b768_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=768 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b768_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b1536_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=1536 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b1536_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b3072_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=3072 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b3072_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b6144_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=6144 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b6144_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b12288_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=12288 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b12288_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5


python pretrain.py name=amp_b128_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=128 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b128_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b256_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=256 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b256_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b384_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=384 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b384_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b768_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=768 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b768_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b1536_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=1536 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b1536_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b3072_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=3072 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b3072_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b6144_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=6144 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b6144_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b12288_c5_o3_br0 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=12288 train.batch_size_ramp=0
python eval.py eval=mnli name=amp_b12288_c5_o3_br0 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# fixed bs300
python pretrain.py name=amp_b96_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=96 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b96_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b192_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=192 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b192_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b384_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=384 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b384_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b768_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=768 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b768_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b1536_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=1536 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b1536_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b3072_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=3072 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b3072_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b6144_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=6144 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b6144_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b12288_c5_o3_br300k wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=12288 train.batch_size_ramp=300000
python eval.py eval=mnli name=amp_b12288_c5_o3_br300k wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5


# ####### Optimizers
#
# fine grained from bert-o3
python pretrain.py name=amp_b4096_c5_o3_sgd wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 ~train.optim.eps train/optim=sgd
python eval.py eval=mnli name=amp_b4096_c5_o3_sgd wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_adamclassic wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train/optim=adam_classic
python eval.py eval=mnli name=amp_b4096_c5_o3_adamclassic wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_adafactor wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 ~train.optim.eps train/optim=adafactor
python eval.py eval=mnli name=amp_b4096_c5_o3_adafactor wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_radam wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train/optim=radam
python eval.py eval=mnli name=amp_b4096_c5_o3_radam wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_lars wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train/optim_mod=lars
python eval.py eval=mnli name=amp_b4096_c5_o3_lars wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_larc wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train/optim_mod=larc
python eval.py eval=mnli name=amp_b4096_c5_o3_larc wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_sam wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train/optim_mod=sam
python eval.py eval=mnli name=amp_b4096_c5_o3_sam wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5


####### Schedulers

# LR variations with schedulers:
python pretrain.py name=amp_b4096_c5_o3_tri2_lr1e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-4
python eval.py eval=mnli name=amp_b4096_c5_o3_tri2_lr1e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_tri2_lr5e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-4
python eval.py eval=mnli name=amp_b4096_c5_o3_tri2_lr5e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_tri2_lr1e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-3
python eval.py eval=mnli name=amp_b4096_c5_o3_tri2_lr1e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_tri2_lr5e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-3
python eval.py eval=mnli name=amp_b4096_c5_o3_tri2_lr5e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5


# cosine
python pretrain.py name=amp_b4096_c5_o3_cosine_lr1e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-4 train.scheduler=budget-cosine-decay
python eval.py eval=mnli name=amp_b4096_c5_o3_cosine_lr1e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_cosine_lr5e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-4 train.scheduler=budget-cosine-decay
python eval.py eval=mnli name=amp_b4096_c5_o3_cosine_lr5e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_cosine_lr1e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-3 train.scheduler=budget-cosine-decay
python eval.py eval=mnli name=amp_b4096_c5_o3_cosine_lr1e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_cosine_lr5e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-3 train.scheduler=budget-cosine-decay
python eval.py eval=mnli name=amp_b4096_c5_o3_cosine_lr5e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# invsqrt
python pretrain.py name=amp_b4096_c5_o3_invsqrt_lr1e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-4 train.scheduler=budget-inverse-sqrt train.warmup_steps=30_000
python eval.py eval=mnli name=amp_b4096_c5_o3_invsqrt_lr1e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_invsqrt_lr5e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-4 train.scheduler=budget-inverse-sqrt train.warmup_steps=30_000
python eval.py eval=mnli name=amp_b4096_c5_o3_invsqrt_lr5e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_invsqrt_lr1e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-3 train.scheduler=budget-inverse-sqrt train.warmup_steps=30_000
python eval.py eval=mnli name=amp_b4096_c5_o3_invsqrt_lr1e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_invsqrt_lr5e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-3 train.scheduler=budget-inverse-sqrt train.warmup_steps=30_000
python eval.py eval=mnli name=amp_b4096_c5_o3_invsqrt_lr5e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# linear
python pretrain.py name=amp_b4096_c5_o3_linear_lr1e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-4 train.scheduler=budget-linear
python eval.py eval=mnli name=amp_b4096_c5_o3_linear_lr1e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_linear_lr5e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-4 train.scheduler=budget-linear
python eval.py eval=mnli name=amp_b4096_c5_o3_linear_lr5e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_linear_lr1e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-3 train.scheduler=budget-linear
python eval.py eval=mnli name=amp_b4096_c5_o3_linear_lr1e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_linear_lr5e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-3 train.scheduler=budget-linear
python eval.py eval=mnli name=amp_b4096_c5_o3_linear_lr5e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# budget-one-cycle
# misnamed, should be budget-one-cycle
python pretrain.py name=amp_b4096_c5_o3_budget_one_cycle wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-4 train.scheduler=budget-one-cycle
python eval.py eval=mnli name=amp_b4096_c5_o3_budget_one_cycle wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_budget_one_cycle_lr5e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-4 train.scheduler=budget-one-cycle
python eval.py eval=mnli name=amp_b4096_c5_o3_budget_one_cycle_lr5e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_budget_one_cycle_lr1e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-3 train.scheduler=budget-one-cycle
python eval.py eval=mnli name=amp_b4096_c5_o3_budget_one_cycle_lr1e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_budget_one_cycle_lr5e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-3 train.scheduler=budget-one-cycle
python eval.py eval=mnli name=amp_b4096_c5_o3_budget_one_cycle_lr5e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# budget-ramp
python pretrain.py name=amp_b4096_c5_o3_ramp_lr1e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-4 train.scheduler=budget-ramp
python eval.py eval=mnli name=amp_b4096_c5_o3_ramp_lr1e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_ramp_lr5e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-4 train.scheduler=budget-ramp
python eval.py eval=mnli name=amp_b4096_c5_o3_ramp_lr5e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_ramp_lr1e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-3 train.scheduler=budget-ramp
python eval.py eval=mnli name=amp_b4096_c5_o3_ramp_lr1e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_ramp_lr5e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-3 train.scheduler=budget-ramp
python eval.py eval=mnli name=amp_b4096_c5_o3_ramp_lr5e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# budget dive
python pretrain.py name=amp_b4096_c5_o3_dive_lr1e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-4 train.scheduler=budget-dive
python eval.py eval=mnli name=amp_b4096_c5_o3_dive_lr1e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_dive_lr5e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-4 train.scheduler=budget-dive
python eval.py eval=mnli name=amp_b4096_c5_o3_dive_lr5e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_dive_lr1e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-3 train.scheduler=budget-dive
python eval.py eval=mnli name=amp_b4096_c5_o3_dive_lr1e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_dive_lr5e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-3 train.scheduler=budget-dive
python eval.py eval=mnli name=amp_b4096_c5_o3_dive_lr5e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# slow dance
python pretrain.py name=amp_b4096_c5_o3_diveslow_lr1e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-4 train.scheduler=budget-dive-slow
python eval.py eval=mnli name=amp_b4096_c5_o3_diveslow_lr1e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_diveslow_lr5e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-4 train.scheduler=budget-dive-slow
python eval.py eval=mnli name=amp_b4096_c5_o3_diveslow_lr5e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_diveslow_lr1e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-3 train.scheduler=budget-dive-slow
python eval.py eval=mnli name=amp_b4096_c5_o3_diveslow_lr1e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_diveslow_lr5e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-3 train.scheduler=budget-dive-slow
python eval.py eval=mnli name=amp_b4096_c5_o3_diveslow_lr5e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# constant
python pretrain.py name=amp_b4096_c5_o3_constant_lr1e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-4 train.scheduler=constant
python eval.py eval=mnli name=amp_b4096_c5_o3_constant_lr1e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_constant_lr5e4 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-4 train.scheduler=constant
python eval.py eval=mnli name=amp_b4096_c5_o3_constant_lr5e4 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_constant_lr1e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=1e-3 train.scheduler=constant
python eval.py eval=mnli name=amp_b4096_c5_o3_constant_lr1e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_constant_lr5e3 wandb.tags=[o3,train] data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 train.optim.lr=5e-3 train.scheduler=constant
python eval.py eval=mnli name=amp_b4096_c5_o3_constant_lr5e3 wandb.tags=[bookcorpus,train] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5
