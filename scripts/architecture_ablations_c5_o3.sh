
# # Basic settings:
python pretrain.py name=amp_b4096_orig_o3 wandb.tags=[o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-original train=bert-o3 impl.microbatch_size=64 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_orig_o3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5
#
python pretrain.py name=amp_b4096_iz_o3 wandb.tags=[o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-large-izsak train=bert-o3 impl.microbatch_size=16 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_iz_o3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c0_o3 wandb.tags=[c0,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-base train=bert-o3 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c0_o3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c2_o3 wandb.tags=[c2,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c2 train=bert-o3 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c2_o3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c3_o3 wandb.tags=[c3,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c3 train=bert-o3 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c3_o3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# Broad architecture ablations
python pretrain.py name=amp_b4096_c5_o3_FFN2 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 arch.ffn_layer_frequency=2 train.steps=2400000
python eval.py eval=mnli name=amp_b4096_c5_o3_FFN2 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_FFN3 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 arch.ffn_layer_frequency=3 train.steps=2400000
python eval.py eval=mnli name=amp_b4096_c5_o3_FFN3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_FFN4 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 arch.ffn_layer_frequency=4 train.steps=2400000
python eval.py eval=mnli name=amp_b4096_c5_o3_FFN4 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5


python pretrain.py name=amp_b4096_c5_o3 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_L4 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.num_transformer_layers=4 train.steps=1200000 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_L4 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_L6 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.num_transformer_layers=6 train.steps=1200000 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_L6 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_L8 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.num_transformer_layers=8 train.steps=1200000 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_L8 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_L10 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.num_transformer_layers=10 train.steps=1200000 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_L10 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_L12 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.num_transformer_layers=12 train.steps=1200000 train.batch_size=4096 impl.microbatch_size=64
python eval.py eval=mnli name=amp_b4096_c5_o3_L12 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_L18 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.num_transformer_layers=18 train.steps=1200000 train.batch_size=4096 impl.microbatch_size=64
python eval.py eval=mnli name=amp_b4096_c5_o3_L18 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_L24 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.num_transformer_layers=24 train.steps=1200000 train.batch_size=4096 impl.microbatch_size=64
python eval.py eval=mnli name=amp_b4096_c5_o3_L24 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_DN_L24 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.steps=1200000 train.batch_size=4096 arch.num_transformer_layers=24 arch.hidden_size=512 arch.intermed_size=2048 arch.attention.num_attention_heads=8 impl.microbatch_size=64 train.steps=1600000
python eval.py eval=mnli name=amp_b4096_c5_o3_DN_L24 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_DN_L12 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.steps=1200000 train.batch_size=4096 arch.num_transformer_layers=12 arch.hidden_size=512 arch.intermed_size=2048 arch.attention.num_attention_heads=8 impl.microbatch_size=64 train.steps=1600000
python eval.py eval=mnli name=amp_b4096_c5_o3_DN_L12 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_large_o3 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.steps=1200000 train.batch_size=4096 arch.num_transformer_layers=24 arch.hidden_size=1024 arch.intermed_size=4096 arch.attention.num_attention_heads=16 impl.microbatch_size=24
python eval.py eval=mnli name=amp_b4096_c5_large_o3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.microbatch_size=8 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_tiny_o3 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.steps=1200000 train.batch_size=4096 arch.num_transformer_layers=2 arch.hidden_size=128 arch.intermed_size=512 arch.attention.num_attention_heads=2 impl.microbatch_size=128 train.steps=3600000
python eval.py eval=mnli name=amp_b4096_c5_tiny_o3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_mini_o3 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.steps=1200000 train.batch_size=4096 arch.num_transformer_layers=4 arch.hidden_size=256 arch.intermed_size=1024 arch.attention.num_attention_heads=4 impl.microbatch_size=128 train.steps=3600000
python eval.py eval=mnli name=amp_b4096_c5_mini_o3 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_H512 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 arch.hidden_size=512 arch.intermed_size=2048 arch.attention.num_attention_heads=8 train.steps=1200000
python eval.py eval=mnli name=amp_b4096_c5_o3_H512 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_H1024 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 arch.hidden_size=1024 arch.intermed_size=4096 arch.attention.num_attention_heads=16 train.steps=1200000 impl.microbatch_size=64
python eval.py eval=mnli name=amp_b4096_c5_o3_H1024 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_E128 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 arch.embedding.embedding_dim=128
python eval.py eval=mnli name=amp_b4096_c5_o3_E128 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_flash wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 arch.attention.type=flash
python eval.py eval=mnli name=amp_b4096_c5_o3_flash base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_fourier wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096 arch.attention.type=fourier
python eval.py eval=mnli name=amp_b4096_c5_o3_fourier base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c2_o3_funnel wandb.tags=[funnel,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=funnel-c2 arch.setup=[128,128,64,64,32,64,64,128,128] train.steps=1200000 train=bert-o3 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c2_o3_funnel base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5 impl.pad_to_multiple_of=128

python pretrain.py name=amp_b4096_c5_o3_rec12 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096  arch.num_transformer_layers=12 arch.recurrent_layers=1
python eval.py eval=mnli name=amp_b4096_c5_o3_rec12 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_rec26 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096  arch.num_transformer_layers=6 arch.recurrent_layers=2
python eval.py eval=mnli name=amp_b4096_c5_o3_rec26 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_rec34 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096  arch.num_transformer_layers=4 arch.recurrent_layers=3
python eval.py eval=mnli name=amp_b4096_c5_o3_rec34 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_rec53 wandb.tags=[c5,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 train.batch_size=4096  arch.num_transformer_layers=3 arch.recurrent_layers=4
python eval.py eval=mnli name=amp_b4096_c5_o3_rec53 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c2_o3_rec12bptt wandb.tags=[funnel,o3,arch] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=recurrent-c2 arch.maximal_recurrence=12 impl.microbatch_size=64 train.steps=1200000
python eval.py eval=mnli name=amp_b4096_c2_o3_rec12bptt base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

# Finegrained ablation of c5_o3:
# python pretrain.py name=amp_b4096_c5_o3_nosparsepred wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.sparse_prediction=False impl.microbatch_size=64 train.batch_size=4096
# python eval.py eval=mnli name=amp_b4096_c5_o3_nosparsepred base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_postln wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.norm_scheme=post train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_postln base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_eps6 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.norm_eps=1e-6 train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_eps6 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_gelu wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.nonlin=GELU train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_gelu base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_decbias wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.decoder_bias=True train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_decbias base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_learnedemb wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.embedding.pos_embedding=learned train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_learnedemb base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_H4 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.attention.num_attention_heads=4 arch.attention.type=self-attention train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_H4 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_H8 wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.attention.num_attention_heads=8 arch.attention.type=self-attention train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_H8 base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_qkvbias wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.attention.qkv_bias=True train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_qkvbias base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_withrot wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.attention.rotary_embedding=True train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_withrot base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_noheadskip wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.skip_head_transform=False train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_noheadskip base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_usebias wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.use_bias=True train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_usebias base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_nofinal wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.final_norm=False train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_nofinal base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5

python pretrain.py name=amp_b4096_c5_o3_noembednorm wandb.tags=[arch,o3] base_dir=/cmlscratch/jonas0/cramming/ impl.local_staging_dir=/scratch0/jonas0 data=bookcorpus-wikipedia arch=bert-c5 train=bert-o3 arch.embedding.normalization=False train.batch_size=4096
python eval.py eval=mnli name=amp_b4096_c5_o3_noembednorm base_dir=/cmlscratch/jonas0/cramming/ wandb.tags=[bookcorpus,arch] eval.checkpoint=latest impl.microbatch_size=16 impl.shuffle_in_dataloader=True eval.scheduler=cosine-decay eval.epochs=5 eval.batch_size=16 eval.optim.lr=4e-5
