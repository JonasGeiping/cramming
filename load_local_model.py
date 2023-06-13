"""Example for a script to load a local saved model.

Use as e.g.

python load_local_model.py name=A6000amp_b4096_c5_o3_final base_dir=~/Documents/cmlscratch_backups/cramming/
> wandb=none impl.push_to_huggingface_hub=True arch=bert-c5 train=bert-o3 train.batch_size=4096
> data=c4-subset-processed dryrun=True +eval=GLUE_sane

"""

import hydra
import time

import logging


import cramming

log = logging.getLogger(__name__)


def main_load_process(cfg, setup):
    """This function controls the central routine."""
    local_time = time.time()

    tokenizer, cfg_arch, model_file = cramming.utils.find_pretrained_checkpoint(cfg)

    model = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=None)
    model_engine, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.train, cfg.impl, setup=setup)
    model_engine.load_checkpoint(cfg_arch, model_file)

    if cramming.utils.is_main_process():
        # Save final checkpoint somewhere else?:
        # now = datetime.datetime.now()
        # checkpoint_id = f"{''.join(cfg.arch.architectures)}_{now.strftime('%Y-%m-%d')}_{float('NaN'):2.4f}"
        # model_engine.save_final_model(os.path.join(cfg.base_dir, cfg.name), checkpoint_id, tokenizer, cfg.arch, cfg.dryrun)

        # Save to hub
        if cfg.impl.push_to_huggingface_hub:
            model_engine.push_to_hub(tokenizer, cfg, dryrun=cfg.dryrun)
    return {}


@hydra.main(config_path="cramming/config", config_name="cfg_pretrain", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_load_process, job_name="load and push model")


if __name__ == "__main__":
    launch()
