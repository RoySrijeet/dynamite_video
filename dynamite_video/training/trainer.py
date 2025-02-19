import os
import copy
import torch
import itertools

from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Set

from detectron2.engine import DefaultTrainer
from detectron2.utils.comm import get_world_size
from detectron2.data.samplers import TrainingSampler
from detectron2.data.common import AspectRatioGroupedDataset, MapDataset
from detectron2.data.build import build_detection_train_loader
from detectron2.solver.build import maybe_add_gradient_clipping

from dynamite_video.data.mappers import TrainingMapper
from dynamite_video.data.utils.collate import Collator
from dynamite_video.data.dataset_builder import build_training_dataset, listify

class Trainer(DefaultTrainer):
    
    @staticmethod
    def get_num_available_cpu_cores() -> int:
        # When running under SLURM, we need to check the `SLURM_CPUS_PER_TASK` environment variable to get the
        # correct number of available CPU cores because multiprocessing.cpu_count() just returns the total number of
        # CPU cores on the machine regardless of how many SLURM has allocated for the given job.
        total_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        return min(8, max(1, total_cores // local_world_size))
    
    
    @classmethod
    def build_train_loader(cls, cfg):
        dataset = build_training_dataset(cfg)

        dataloader = build_detection_train_loader(cfg, 
                                                  dataset=dataset, 
                                                  mapper=TrainingMapper(cfg),
                                                  #collate_fn=Collator(cfg, is_train=True)
                                            )
        return dataloader    
    
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    #print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer
    