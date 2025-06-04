import json
import os
import wandb


PARAM_NAME_ALIAS = {
    "CLICKER.TRAINING.MAX_NUM_INSTANCES_REFINED_PER_ROUND": "inst",
    "MODEL.MASK_FORMER.QQCA": "qqca",
    "SOLVER.BASE_LR": "lr",
    "SOLVER.MAX_ITERS": "iters"
}

def listify(dct):
    lst = []
    suffix = ""
    for k,v in dct.items():
        suffix += "_" + PARAM_NAME_ALIAS[k] + "_" + str(v)
        lst.extend([k,v])
    return lst, suffix


def wandb_init(cfg):
    
    wandb.tensorboard.unpatch()
    
    # initialize
    wandb.init(entity=cfg.WANDB.ENTITY, 
                project=cfg.WANDB.PROJECT
            )

    suffix = "" 
    if cfg.WANDB.SWEEP:
        curr_sweep_params, suffix = listify(wandb.config)
        # inject sweep parameters into cfg
        cfg.defrost()
        cfg.merge_from_list(curr_sweep_params)
        cfg.WANDB.RUN_NAME = cfg.WANDB.RUN_NAME + suffix
        cfg.freeze()
    
    wandb.tensorboard.patch(root_logdir=os.path.join(cfg.OUTPUT_DIR, cfg.WANDB.RUN_NAME))
    
    wandb.config.update(cfg, allow_val_change=True)
    wandb.run.name = cfg.WANDB.RUN_NAME


def wandb_sweep(args, cfg, launch_fn):
    # read sweep parameters
    path_to_sweep = os.path.join(args.expt_dir, "sweep.json")
    assert os.path.exists(path_to_sweep), \
        f"args.expt_dir ({args.expt_dir}) must contain `sweep.json`"
    
    with open(path_to_sweep, "r") as f:
        sweep_config = json.load(f)
    
    # create a sweep
    sweep_id = wandb.sweep(sweep_config, project=cfg.WANDB.PROJECT)
    wandb.agent(sweep_id, function=lambda: launch_fn(args))
    return