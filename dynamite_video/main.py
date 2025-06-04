import os
from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append(os.environ["DYNAMITE_VIDEO_WORKSPACE"])

import wandb
import torch
import imgaug
import random
import numpy as np

from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer

from dynamite_video.training.trainer import Trainer
from dynamite_video.evaluation.evaluator import Evaluator
from dynamite_video.utils.misc import get_cl_arguments, load_config
from dynamite_video.utils.wandb import wandb_init, wandb_sweep


def seed_rngs(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    imgaug.seed(seed)


def evaluation_pipeline(cfg, args):
    logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name=__name__)
    logger.info("Welcome to DynaMITe-Video Evaluation Pipeline!")

    # load model architecture
    model = Evaluator.build_model(cfg)
    # load model weights from cfg.MODEL.WEIGHTS
    DetectionCheckpointer(model, 
                          save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS)
    
    Evaluator.interactive_evaluation(cfg, args, model)


def training_pipeline(cfg):
    # set seeds manually
    seed_rngs(cfg.SEED)

    logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name=__name__)
    logger.info("Welcome to DynaMITe-Video Training Pipeline!")

    trainer = Trainer(cfg)

    # W&B
    if cfg.WANDB.ENABLE and comm.get_rank()==0:
        wandb_init(cfg, trainer.model)
        
    trainer.resume_or_load(cfg.TRAINING.RESUME)
    trainer.train()


def main(args):

    # setup experiment configuration
    cfg = load_config(args)

    if args.eval_only:
        evaluation_pipeline(cfg, args)
        return
    
    # W&B sweep
    if cfg.WANDB.SWEEP and comm.get_rank() == 0:
        wandb_sweep(sweep_config=os.path.join(args.expt_dir, "sweep.json"), 
                    cfg=cfg, launch_fn=training_pipeline
                )
    
    # just a regular run
    training_pipeline(cfg)


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