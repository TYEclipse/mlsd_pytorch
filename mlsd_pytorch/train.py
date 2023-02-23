import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.optim
import yaml

sys.path.append(os.path.dirname(__file__)+'/../')

from mlsd_pytorch.cfg.default import get_cfg_defaults
from mlsd_pytorch.data import get_train_dataloader, get_val_dataloader
from mlsd_pytorch.learner import Simple_MLSD_Learner
from mlsd_pytorch.models.build_model import build_model
from mlsd_pytorch.optim.lr_scheduler import WarmupMultiStepLR
from mlsd_pytorch.utils.comm import create_dir, setup_seed
from mlsd_pytorch.utils.logger import TxtLogger


TRAIN_LOGGER_FILENAME = "train_logger.txt"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent /
                        "configs" / "mobilev2_mlsd_tiny_512_base.yaml", help="")
    return parser.parse_args()


def train(cfg):
    train_loader = get_train_dataloader(cfg)
    val_loader = get_val_dataloader(cfg)
    model = build_model(cfg).cuda()

    if os.path.exists(cfg.train.load_from):
        print('load from: ', cfg.train.load_from)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(
            cfg.train.load_from, map_location=device), strict=False)

    if cfg.train.milestones_in_epo:
        ns = len(train_loader)
        milestones = [m * ns for m in cfg.train.milestones]
        cfg.train.milestones = milestones

    optimizer = torch.optim.Adam(params=model.parameters(
    ), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    if cfg.train.use_step_lr_policy:
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=cfg.train.milestones,
            gamma=cfg.train.lr_decay_gamma,
            warmup_iters=cfg.train.warmup_steps,
        )
    else:  # similiar with in the paper
        warmup_steps = 5 * len(train_loader)  # 5 epoch warmup
        min_lr_scale = 0.0001
        start_step = 70 * len(train_loader)
        end_step = 150 * len(train_loader)
        n_t = 0.5

        def lr_lambda_fn(step):
            if step < warmup_steps:
                return 0.9 * step / warmup_steps + 0.1
            elif step < start_step:
                return 1.0
            elif n_t * (1 + math.cos(math.pi * (step - start_step) / (end_step - start_step))) < min_lr_scale:
                return min_lr_scale
            else:
                return n_t * (1 + math.cos(math.pi * (step - start_step) / (end_step - start_step)))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda_fn)

    save_dir = cfg.train.save_dir
    logger = TxtLogger(os.path.join(save_dir, TRAIN_LOGGER_FILENAME))
    learner = Simple_MLSD_Learner(
        cfg,
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        logger=logger,
        save_dir=save_dir,
        log_steps=cfg.train.log_steps,
        device_ids=cfg.train.device_ids,
        gradient_accum_steps=1,
        max_grad_norm=1000.0,
        batch_to_model_inputs_fn=None,
        early_stop_n=cfg.train.early_stop_n)

    learner.train(train_loader, val_loader, epoches=cfg.train.num_train_epochs)

    score_threshold = cfg.decode.score_thresh
    top_k = cfg.decode.top_k
    length_threshold = cfg.decode.len_thresh

    print('iter of train_loader: ', len(train_loader))
    learner.val(model, train_loader, score_threshold, top_k, length_threshold)
    print('iter of val_loader: ', len(val_loader))
    learner.val(model, val_loader, score_threshold, top_k, length_threshold)


def save_cfg(cfg):
    create_dir(cfg.train.save_dir)
    with open(cfg.train.save_dir + "/cfg.yaml", "w") as f:
        yaml.dump(cfg, f)

    print(f'saved config to {cfg.train.save_dir}/cfg.yaml')


if __name__ == '__main__':
    # Set up seed for reproducibility
    setup_seed(6666)

    # Load configuration from file
    cfg = get_cfg_defaults()
    args = get_args()
    cfg.merge_from_file(args.config)

    # Print and save configuration
    print(f'Using config: {args.config}')
    print(cfg)
    save_cfg(cfg)

    # Train the model
    train(cfg)

# CUDA_VISIBLE_DEVICES=4 python mlsd_pytorch/train.py --config mlsd_pytorch/configs/mobilev2_mlsd_large_512_base2_bsize16_LSDdataset_only_link.yaml
# CUDA_VISIBLE_DEVICES=2 python mlsd_pytorch/train.py --config mlsd_pytorch/configs/mobilev2_mlsd_large_1280_base2_bsize2_LSDdataset_only_link.yaml
