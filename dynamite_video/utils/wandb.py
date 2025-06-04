import json
import wandb


def listify(dct):
    lst = []
    return [lst.extend([k,v]) for k,v in dct.items()]


def wandb_init(cfg, model):
    
    # initialize
    wandb.init(entity=cfg.WANDB.ENTITY, 
                project=cfg.WANDB.PROJECT, 
                sync_tensorboard=True
            )

    suffix = "2" 
    if cfg.WANDB.SWEEP:
        # inject sweep parameters into cfg
        cfg.merge_from_list(listify(wandb.config))
        cfg.freeze()
        # set a dynamic run name using sweep values
    
    wandb.config.update(cfg, allow_val_change=True)
    run_name = cfg.WANDB.RUN_NAME + suffix
    wandb.run.name = run_name
    
    if cfg.WANDB.WATCH_GRAD:
        # watch parameter gradients
        wandb.watch(model, log="all", log_freq=10)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")



def wandb_sweep(sweep_config, cfg, launch_fn):
    with open(sweep_config, "r") as f:
        sweep_config = json.load(f)
    sweep_id = wandb.sweep(sweep_config, project=cfg.WANDB.PROJECT)
    wandb.agent(sweep_id, function=lambda: launch_fn(cfg))
    return