import os
import sys
import argparse

from datetime import datetime
import torch.distributed as dist

from dynamite_video.utils.paths import Paths
from dynamite_video.utils.detectron2_custom import default_setup
from dynamite_video.utils.config import add_maskformer2_config

from detectron2.utils import comm
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config # type: ignore


def get_expt_config(expt_dir: str):
    """
    Path to the configuration file in experiment directory.
    Experiment configuration modifies the base configuration.
    """
    yaml_files = []
    try:
        for file in os.listdir(expt_dir):
            if file.endswith(".yaml"):
                # detectron2 saves the updated config file as config.yaml, ignore this file
                if file=="config.yaml":
                    continue
                yaml_files.append(os.path.join(expt_dir, file))
    except:
        raise RuntimeError(f"Experiment configuration could not be found at {expt_dir}!")

    assert len(yaml_files)==1, f"Experiment directory ({expt_dir}) must contain 1 yaml file, found {len(yaml_files)}. \
                                File *config.yaml* is ignored as detectron2 uses this name to save updated config."
    return yaml_files[0]


def get_base_config():
    """
    Path to base configuration file
    """
    return os.path.join(Paths.to_configs(), "base.yaml")


def load_config(args):
    """
    Load experiment configurations

    First load base configurations. For a fresh run, load current experiment
    configurations from experiment folder (`args.expt_dir`) and update base 
    configurations. Optional command line arguments (`args.opts`) may also 
    alter the config. Output directory is set to the experiment folder. This 
    is where logs, checkpoints and back-ups are saved.

    If resuming a previous training run whose checkpoint path is specified by 
    `args.resume`, ensure the checkpoint directory also contains the configurations
    of the previous run (`config.yaml`). This configuration is used to update
    the base configuration. Outputs of resumed training are saved in the experiment
    folder specified by `args.expt_dir`.

    Note that in resumed training, any custom configuration file stored in the 
    experiment directory is ignored and so is any optional arguments (`args.opts`) 
    passed through the command line.

    Args:
        args: command line arguments - only interested in args.opts here
        expt_config: path to the directory containing experiment .yaml file

    """
    # detectron2 base config
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    
    # load base configuration
    base_cfg = get_base_config()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(base_cfg)
    # print(cfg)

    if args.resume or args.eval_only:
        # check if checkpoint directory contains config file
        ckpt_folder = args.expt_dir #os.path.dirname(args.resume) if args.resume else args.expt_dir
        path_to_ckpt_config = os.path.join(ckpt_folder, "config.yaml")
        assert os.path.exists(path_to_ckpt_config), f"Config file not found! Checkpoint folder \
            ({ckpt_folder}) must contain both .pth and config.yaml files from previous run."
        
        # update with checkpoint configuration
        cfg.merge_from_file(path_to_ckpt_config)
        if args.eval_config:
            cfg.set_new_allowed(True)
            cfg.merge_from_file(args.eval_config)

    else:
        # experiment config
        cfg.merge_from_file(get_expt_config(args.expt_dir))
    
    
    # outputs are saved in experiment directory
    cfg.OUTPUT_DIR = args.expt_dir
    
    # during evaluation, create a sub-directory inside expt_dir to save outputs
    if args.eval_only:
        output_dir = os.path.join(args.expt_dir, "output")
        if os.path.isdir(output_dir):
            output_dir = output_dir + datetime.now().strftime('%Y_%m_%d_%H%M%S')
        os.makedirs(output_dir)
        cfg.OUTPUT_DIR = output_dir

    # command line overwrites
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="dynamite-video")
    return cfg


def get_cl_arguments():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(
        prog="DynaMITe-Video",
        epilog=f"""
                Getting started:

                Run on single machine:
                    $ {sys.argv[0]} --num-gpus 4 --expt-dir path/to/experiment/folder

                Change some config options:
                    $ {sys.argv[0]} --expt-dir path/to/experiment/folder MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

                Run on multiple machines:
                    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
                    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
                """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    
    parser.add_argument(           # eval
        "--eval-config", 
        type=str,
        default=None,
        help="custom evaluation config"
    )
    
    parser.add_argument(           # eval
        "--eval-datasets", 
        type=tuple_type, 
        help="perform evaluation on given datsets"
    )
    
    parser.add_argument(           # eval
        "--eval-only", 
        action="store_true", 
        help="perform evaluation only"
    )

    parser.add_argument(            # eval
        "--eval-strategy", 
        type=str, 
        default="random", 
        help="Strategy to select the instance to add corrective clicks on. Default: 'random'"
        "Choose between 'random', 'worst', 'best'."
    )

    parser.add_argument(
        "--expt-dir",
        required=True, 
        type=str,
        help="output path; directory where checkpoints and visualizations are to be "
        "saved. Directory must contain one experiment configuration file (.yaml)."
    )

    parser.add_argument(
        "--finetune", 
        type=str,
        required=False,
        help="Use this to start the finetuning training after pre-training on image datasets. "
        "In addition to loading model weights, this also loads the config from the pre-trained "
        "checkpoint directory before overwriting it with the given '--cfg'. This option should point "
        "to the directory containing the pretrained checkpoint, and not the checkpoint file itself."
    )

    parser.add_argument(           # eval
        "--iou-threshold",
        type=float, 
        default=0.85,
        help="IoU threshold for interactive evaluation"
    )

    parser.add_argument(
        "--machine-rank", 
        type=int, 
        default=0, 
        help="Unique rank of this machine (machine==node)"
    )

    parser.add_argument(            # eval
        "--max-interactions",
        type=int, 
        default=10,
        help="Max no. of interactions allowed per instance for interactive evaluation"
    )

    parser.add_argument(            # eval
        "--max-rounds",
        type=int, 
        default=3,
        help="Max no. of corrective rounds in interactive evaluation"
    )

    parser.add_argument(
        "--num-gpus", 
        type=int, 
        default=1, 
        help="Number of GPUs *per machine* (machine==node)"
    )
    
    parser.add_argument(
        "--num-machines", 
        type=int, 
        default=1, 
        help="Total number of machines (machine==node)"
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="""
                Modify config options at the end of the command. For Yacs configs, use
                space-separated "PATH.KEY VALUE" pairs.
                For python-based LazyConfig, use "path.key=value".
                """.strip(),
    )

    parser.add_argument(
        "--resume", 
        action="store_true",
        help="""
                Whether to resume from specified checkpoint directory. Resuming means loading all available 
                states (eg. optimizer and scheduler) and update iteration counter from the checkpoint.
                If true and cfg.OUTPUT_DIR contains the last checkpoint (defined by a last_checkpoint file), 
                resume from the file. cfg.MODEL.WEIGHTS will not be used.
                If false, run an independent training. Load weights from cfg.MODEL.WEIGHTS and start from iter 0.
            """
    )

    parser.add_argument(
        "--seed-id", 
        type=int, 
        default=0, 
        help="Seed id for random evaluation."
    )

    parser.add_argument(           # eval
        "--save-vis", 
        action="store_true", 
        help="Save visualizations of predictions."
    )

    args = parser.parse_args()
    
    # check that experiment directory exists
    assert os.path.isdir(args.expt_dir), f"Path to output directory {args.expt_dir} \
         does not exist"

    return args


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    return tuple(strings.split(","))

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True