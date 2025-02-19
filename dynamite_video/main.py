import os
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append(os.environ["DYNAMITE_VIDEO_WORKSPACE"])

import cv2
import yaml
import wandb
import torch
import imgaug
import random
import logging
import numpy as np
from datetime import datetime, timedelta

from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer

from dynamite_video.training.trainer import Trainer
from dynamite_video.utils.misc import get_cl_arguments, load_config


def seed_rngs(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    imgaug.seed(seed)


def training_pipeline(cfg, args):
    print("Welcome to DynaMITe-Video Training Pipeline!")

    # set seeds manually
    seed_rngs(args.seed_id)

    # W&B
    if cfg.WANDB.ENABLE:
        wandb.init(entity=cfg.WANDB.ENTITY, 
                   project=cfg.WANDB.PROJECT, 
                   name=cfg.WANDB.RUN_NAME, 
                   config=cfg,
                   sync_tensorboard=True
        )

    trainer = Trainer(cfg)
    trainer.resume_or_load(args.resume)
    trainer.train()


def inference_pipeline(cfg, args):
    print("Welcome to DynaMITe-Video Evaluation Pipeline!")

    # load model architecture
    model = Trainer.build_model(cfg)
    # load model weights from cfg.MODEL.WEIGHTS
    DetectionCheckpointer(model, 
                          save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS)
    


def main(args):

    # setup experiment configuration
    cfg = load_config(args)

    if args.eval_only:
        inference_pipeline(cfg, args)
    else:
        training_pipeline(cfg, args)


if __name__=="__main__":

    # read command line arguments
    args = get_cl_arguments()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )