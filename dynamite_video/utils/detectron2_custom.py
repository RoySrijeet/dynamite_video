# Code snippets from detectron2 (v6) library with modifications
# to remove deprecated code, warnings, and minor inconsistencies.

import os
import torch
from omegaconf import OmegaConf

from detectron2.config import CfgNode, LazyConfig
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detectron2.utils.collect_env import collect_env_info


__all__ = [
    "default_setup",
]

def _set_float32_precision(precision: str = "high") -> None:
    """Sets the precision of float32 matrix multiplications and convolution operations.

    For more information, see the PyTorch docs:
    - https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    - https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.allow_tf32

    Args:
        precision: The setting to determine which datatypes to use for matrix
        multiplication and convolution operations.
    """
    if not (torch.cuda.is_available()):  # Not relevant for non-CUDA devices
        return
    # set precision for matrix multiplications
    torch.set_float32_matmul_precision(precision)
    # set precision for convolution operations
    if precision == "highest":
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True


def _try_get_key(cfg, *keys, default=None):
    """
    Try select keys from cfg until the first key that exists. Otherwise return default.
    """
    if isinstance(cfg, CfgNode):
        cfg = OmegaConf.create(cfg.dump())
    for k in keys:
        none = object()
        p = OmegaConf.select(cfg, k, default=none)
        if p is not none:
            return p
    return default


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    if cfg.WANDB.SWEEP:
        # if the run is part of a sweep, create a subdirectory
        cfg.defrost()
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.WANDB.RUN_NAME)
        cfg.freeze()
    
    output_dir = cfg.OUTPUT_DIR

    if comm.is_main_process() and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rank = comm.get_rank()
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))

    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        if isinstance(cfg, CfgNode):
            # logger.info("Running with full config:\n{}".format(cfg.dump()))
            with open(path, "w") as f:
                f.write(cfg.dump())
        else:
            LazyConfig.save(cfg, path)
        logger.info("Full config saved to {}".format(path))

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )

    fp32_precision = _try_get_key(cfg, "FLOAT32_PRECISION", "train.float32_precision", default="")
    if fp32_precision != "":
        logger.info(f"Set fp32 precision to {fp32_precision}")
        _set_float32_precision(fp32_precision)
        logger.info(f"{torch.get_float32_matmul_precision()=}")
        logger.info(f"{torch.backends.cuda.matmul.allow_tf32=}")
        logger.info(f"{torch.backends.cudnn.allow_tf32=}")