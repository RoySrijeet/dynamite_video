import os
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.structures import ImageList

from dynamite_video.model.loss.criterion import SetFinalCriterion


@META_ARCH_REGISTRY.register()
class DynamiteModel(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
    @configurable
    def __init__(
        self,
        *,
        backbone: nn.Module,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        iterative_evaluation: bool
    ):
        """
        Args:
            backbone: a backbone module
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            interactive_evaluation: bool to indicate if it's just one time inference or iterative evaluation
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # iterative
        self.iterative_evaluation = iterative_evaluation


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION

        # loss weights
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
     
        # building criterion
        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight}
        losses = ["masks"]

        criterion = SetFinalCriterion(
            weight_dict=weight_dict,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        if deep_supervision:
            enc_layers = cfg.MODEL.MASK_FORMER.ENC_LAYERS
            aux_weight_dict = {}
            for i in range(enc_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,

            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,

            #iterative
            "iterative_evaluation": cfg.ITERATIVE.TEST.INTERACTIVE_EVALAUTION,
        }
        
    
    @property
    def device(self):
        return self.pixel_mean.device

    
    def forward(
            self, 
            inputs
    ):
        """
        Forward pass through the DynaMITe model
        """

        assert len(inputs) == 1, "Don't try more than one clip in a batch"     
        
        # extract resources from batch
        (images, 
        num_clicks_per_target,
        fg_coords, 
        bg_coords, 
        max_timestamp
        ) = self.preprocess_batch_data(inputs)

        # extract backbone features from clip frames
        features = self.backbone(images.tensor)
        
        if self.training:
        
            # prepare ground truth mask information
            targets = self.prepare_targets(inputs)
            
            # forward to pixel decoder and interactive transformer
            outputs, num_queries_per_target = self.sem_seg_head(inputs[0], 
                                                                images,
                                                                features,
                                                                num_clicks_per_target,
                                                                fg_coords,
                                                                bg_coords,
                                                                max_timestamp)
            # loss computation
            losses = self.criterion(outputs, targets, num_queries_per_target)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
           
        else: # evaluation
            outputs, num_queries_per_target, queries, normalized_clicks = self.sem_seg_head(inputs[0],
                                                                                            images,
                                                                                            features,
                                                                                            num_clicks_per_target,
                                                                                            fg_coords, 
                                                                                            bg_coords, 
                                                                                            max_timestamp)
            return images, outputs, num_queries_per_target, queries, normalized_clicks


    def preprocess_batch_data(self, inputs):
        """
        Given a batch of clips, extract images as `torch.Tensor` as well as
        initial click coordinates and click count.

        Returns:
            images: (d2) ImageList targets, contains the image tensors of the frames in 
                    the corresponding clip as [T,3,H,W] tensors, where T: #frames in the clip
            num_clicks_per_target: list of click count per target
            fg_coords: list of foreground clicks
            bg_coords: list of background clicks
        """
        clip = inputs[0]
        
        # convert each frame in the clip to `torch.Tensor`
        images_sample = [x.to(self.device) for x in clip["images"]]
        # normalize each frame
        images_sample = [(x - self.pixel_mean) / self.pixel_std for x in images_sample]
        # store the frames as detectron2.ImageList
        images_sample = ImageList.from_tensors(images_sample, self.size_divisibility)
        images = images_sample
        
        # extract target and click info
        num_clicks_per_target = clip["num_clicks_per_object"]
        fg_coords = clip["fg_coords_list"]
        bg_coords = clip["bg_coords_list"]
        max_timestamp = clip["max_timestamp_list"]

        return images, num_clicks_per_target, fg_coords, bg_coords, max_timestamp


    def prepare_targets(self, inputs):
        """
        Extract ground truth masks and labels of the targets. Relevant only in the training.

        Args:
            inputs: batch

        Returns:
            A list of dictionaries, one for each frame in the clip. Each dict contains:
                * binary_masks - ground truth binary masks of target targets (N,H,W)
                * semantic_masks - panoptic mask of each frame (H,W)
                * bg_mask - background mask of each frame (H,W)
                * labels - labels of the targets in the clip (a list of ints)
                * padding_mask - padding applied to the clip (H,W)
                * ignore_mask - ignore mask of each frame (H,W)
        """

        targets = []
        clip = inputs[0]
        for i in range(clip["images"].shape[0]):
            targets.append({
                "binary_masks": clip["binary_masks"][i].to(self.device),
                "padding_mask": clip["padding_mask"].to(self.device),
                "ignore_masks": clip["ignore_masks"][i].to(self.device) if clip["ignore_masks"] is not None else None, 
                "panoptic_masks": clip["panoptic_masks"][i].to(self.device),
                "bg_masks": clip["bg_masks"][i].to(self.device) if clip["bg_masks"] is not None else None,
            })
        return targets