import copy
import itertools
import torch

from torch.utils.data import DataLoader
from typing import Any, Dict, List, Set

from detectron2.engine import DefaultTrainer
from detectron2.solver.build import maybe_add_gradient_clipping

from dynamite_video.data.dataset_builder import build_training_dataset
from dynamite_video.data.utils.collate import collate_fn_pretrain

class Trainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build training dataset and return training data loader.
        """
        dataset = build_training_dataset(cfg)
        
        return DataLoader(
                dataset,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                collate_fn=collate_fn_pretrain,
            )

        
        # from detectron2.data.build import build_batch_data_loader
        # from detectron2.data.common import DatasetFromList, MapDataset
        # from detectron2.data.samplers import TrainingSampler
        # from dynamite_video.data.mappers import TrainingMapper
        # if cfg.TRAINING.PRETRAIN:
        #     return DataLoader(
        #         dataset,
        #         batch_size=cfg.SOLVER.IMS_PER_BATCH,
        #         num_workers=cfg.DATALOADER.NUM_WORKERS,
        #         collate_fn=collate_fn_pretrain,
        #     )
        
        # # NOTE: skipping serializing samples to avoid OOM 
        # dataset = DatasetFromList(dataset, copy=False, serialize=False)
        # # training map function over the elements in the dataset
        # # this converts the clip metadata into the actual clip used in training
        # # mapper = TrainingMapper(cfg)
        # # dataset = MapDataset(dataset, mapper)

        # return build_batch_data_loader(
        #                 dataset,
        #                 sampler=TrainingSampler(len(dataset), seed=cfg.SEED),
        #                 total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
        #                 aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        #                 num_workers=cfg.DATALOADER.NUM_WORKERS,
        #             )

    
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
    