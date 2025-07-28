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


from dynamite_video.training.trainer import Trainer
from dynamite_video.evaluation.evaluator import Evaluator
from dynamite_video.utils.detectron2_custom import default_setup
from dynamite_video.utils.misc import get_cl_arguments, load_config
from dynamite_video.utils.wandb import wandb_init, wandb_sweep


def seed_rngs(seed: int):
    imgaug.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(str(seed))


def evaluation_pipeline(args):

    # setup experiment configuration
    cfg = load_config(args)
    default_setup(cfg, args)

    evaluator = Evaluator(cfg)
    evaluator.interactive_evaluation()


def training_pipeline(args):
    
    cfg = load_config(args)
    
    seed_rngs(cfg.SEED)

    # W&B
    if cfg.WANDB.ENABLE and comm.get_rank()==0:
        wandb_init(cfg)

    default_setup(cfg, args)
    logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name=__name__)
    logger.info("Welcome to DynaMITe-Video Training Pipeline!")

    trainer = Trainer(cfg)

    # watch parameter gradients
    if cfg.WANDB.ENABLE and cfg.WANDB.WATCH_GRAD:
        wandb.watch(trainer.model, log="all", log_freq=cfg.SOLVER.CHECKPOINT_PERIOD//2)
        
    trainer.resume_or_load(cfg.TRAINING.RESUME)
    trainer.train()


def main(args):

    if args.eval_only:
        evaluation_pipeline(args)
        return
    
    # setup experiment configuration
    cfg = load_config(args)
    
    # W&B sweep
    if cfg.WANDB.SWEEP and comm.get_rank() == 0:
        wandb_sweep(args, cfg, launch_fn=training_pipeline)
    
    # just a regular run
    training_pipeline(args)


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