import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import torch
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_tf32 = True

import hydra
from omegaconf import OmegaConf
import pathlib, os
import time

from diffusion_policy import DIFFUSION_POLICY_ROOT
from diffusion_policy.trainer.base_trainer import BaseTrainer

OmegaConf.register_new_resolver("eval", eval, replace=True)

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

# diffusion policy arguments
parser.add_argument("--cfg", required=True, help="Name of the config file")
parser.add_argument("--exp_name", required=False, default='default_exp_name', help="exp name")

# parse the arguments
args_cli = parser.parse_args()

def main():
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    cfg: OmegaConf = OmegaConf.load(os.path.join(DIFFUSION_POLICY_ROOT, './config_files', args_cli.cfg))
    OmegaConf.resolve(cfg)

    cfg.exp_name = args_cli.cfg[:-5]
    
    cfg.output_dir = os.path.join(cfg.output_dir, time.strftime("%B-%d-%H-%M-%S", time.localtime()) + "-" + cfg.exp_name)
    os.makedirs(cfg.output_dir, exist_ok=True)

    cls = hydra.utils.get_class(cfg._target_)
    trainer: BaseTrainer = cls(cfg)
    
    trainer.train(args_cli)

if __name__ == "__main__":
    main()