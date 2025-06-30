# Adapted from https://github.com/amitrana001/DynaMITe

import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

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
        iterative_evaluation: bool,
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
        #sem_seg_head = DynamiteHead(cfg, backbone.output_shape())
        
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
            inputs, 
            images=None,
            objects_per_frame=None,
            features=None, 
            mask_features=None, 
            multi_scale_features=None, 
            num_clicks_per_object= None,
            fg_coords = None, 
            bg_coords = None, 
            max_timestamp=None,
            visualize=None,
            train_iter=None
    ):
        """
        Forward pass through the DynaMITe model

        Args:
            inputs: a list, batch output from DataLoader
            features, mask_features, multi_scale_features:
                computed once per clip and passed as an argument to avoid 
                recomputation during iterative evaluation/inference
            fg_coords: batched per clip, list of clicks coordinates for each object in each frame
            bg_coords: batched per clip, list of background click coordinates from each frame
            num_clicks_per_object: batched per clip, number of clicks per object in each frame
            max_timestamp: batched per clip, timestamp of last click sampled from each clip
            objects_per_frame: batched per clip, list of objects present in the frame
        """

        assert len(inputs) == 1, "Don't try more than one clip in a batch"  # TODO
        
        # extract resources from batch
        (images, 
        objects_per_frame, 
        num_clicks_per_object,
        fg_coords, 
        bg_coords, 
        max_timestamp
        ) = self.preprocess_batch_data(inputs)

        if features is None:
            # extract backbone features from clip frames
            clip_fs = self.backbone(images.tensor)
            features = clip_fs

        
        if self.training:
        
            # prepare ground truth mask information
            targets = self.prepare_targets(inputs)

            if visualize:
                import os
                visualize_dir = "/home/roy/REPOS/dynamite_video/visualization/inputs"
                torch.save(images, os.path.join(visualize_dir, f"images_iter_{train_iter}.pth"))
                torch.save(features, os.path.join(visualize_dir, f"features_iter_{train_iter}.pth"))
                torch.save(targets, os.path.join(visualize_dir, f"targets_iter_{train_iter}.pth"))
                torch.save(inputs, os.path.join(visualize_dir, f"inputs_iter_{train_iter}.pth"))
            
            outputs, num_queries_per_object = self.sem_seg_head(inputs[0], 
                                                                images,
                                                                features,
                                                                objects_per_frame,
                                                                mask_features, 
                                                                multi_scale_features,
                                                                num_clicks_per_object,
                                                                fg_coords,
                                                                bg_coords,
                                                                max_timestamp,
                                                                visualize=visualize,
                                                                train_iter=train_iter,
                                                            )
            
            losses = self.criterion(outputs, targets, num_queries_per_object, visualize=visualize, train_iter=train_iter)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
           
        else:
            # iterative evaluation - for each batch (a clip) we only compute image features and 
            # mask features once and pass them as arguments to use them again in the next round

            (outputs, num_queries_per_object, queries,
            mask_features, multi_scale_features) = self.sem_seg_head(inputs[0],
                                                                    images,
                                                                    features,
                                                                    objects_per_frame,
                                                                    mask_features, 
                                                                    multi_scale_features, 
                                                                    num_clicks_per_object,
                                                                    fg_coords, 
                                                                    bg_coords, 
                                                                    max_timestamp
                                                                )
            
            processed_results, queries = self.process_results(images, outputs, queries, objects_per_frame, num_queries_per_object)
            
            return processed_results, queries, num_queries_per_object


    def preprocess_batch_data(self, inputs):
        """
        Given a batch of clips, extract images as `torch.Tensor` as well as
        initial click coordinates and click count.

        Returns:
            images: (d2) ImageList objects, contains the image tensors of the frames in 
                    the corresponding clip as [T,3,H,W] tensors, where T: #frames in the clip
            objects_per_frame: list of list of object IDs in each frame of the clip
            num_clicks_per_object: list of click count per object
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
        
        # extract object and click info
        objects_per_frame = clip["objects_per_frame"]
        num_clicks_per_object = clip["num_clicks_per_object"]
        fg_coords = clip["fg_coords_list"]
        bg_coords = clip["bg_coords_list"]
        max_timestamp = clip["max_timestamp_list"]

        return images, objects_per_frame, num_clicks_per_object, fg_coords, bg_coords, max_timestamp


    def prepare_targets(self, inputs):
        """
        Extract ground truth masks and labels of the objects. Relevant only in the training.

        Args:
            inputs: batch

        Returns:
            A list of dictionaries, one for each frame in the clip. Each dict contains:
                * binary_masks - ground truth binary masks of target objects (N,H,W)
                * semantic_masks - panoptic mask of each frame (H,W)
                * bg_mask - background mask of each frame (H,W)
                * labels - labels of the objects in the clip (a list of ints)
                * padding_mask - padding applied to the clip (H,W)
                * ignore_mask - ignore mask of each frame (H,W)
        """

        targets = []
        clip = inputs[0]
        for i in range(clip["images"].shape[0]):
            targets.append({
                "binary_masks": clip["binary_masks"][i].to(self.device),
                "semantic_masks": clip["semantic_masks"][i].to(self.device),
                "bg_masks": clip["bg_masks"][i].to(self.device) if clip["bg_masks"]is not None else None,
                "labels": clip["objects_per_frame"][i],
                "padding_mask": clip["padding_mask"].to(self.device),
                "ignore_masks": clip["ignore_masks"][i].to(self.device) if clip["ignore_masks"] is not None else None, 
            })
        return targets


    ### Evaluation ###
    def process_results(
            self, 
            images, 
            outputs,
            queries, 
            objects_per_frame,
            num_queries_per_object,
    ):
        """
        Args:
            images: [T, 3, H, W] tensors of the images in the clip (d2 ImageList)
            outputs: prediction 
            objects_per_frame: list of object IDs in the i-th frame
            num_queries_per_object: count of queries on each object in each frame
        """
        
        mask_pred_results = outputs["pred_masks"]   # [T,Q,H,W]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del outputs

        # objects in the whole clip
        seq_objects = sorted(objects_per_frame)

        processed_results = []
        processed_queries = []
        for mask_pred_per_image, queries_per_image, image_size in zip(mask_pred_results, queries, images.image_sizes):
            mask_pred_per_image = retry_if_cuda_oom(sem_seg_postprocess)(mask_pred_per_image, image_size, image_size[0], image_size[1])
            processed_r, queries_r = retry_if_cuda_oom(self.interactive_object_inference)(mask_pred_per_image, 
                                                                               queries_per_image,
                                                                               num_queries_per_object, 
                                                                               seq_objects)
            processed_results.append(processed_r)
            processed_queries.append(queries_r)

        return processed_results, torch.stack(processed_queries)

    
    def interactive_object_inference(
            self, 
            mask_pred, 
            queries,
            num_queries_per_object,
            seq_objects,
    ):
        """
        Given the raw predictions from Transformer, obtain binary segmentation masks

        Args:
            mask_pred: raw prediction from Transformer, TxQxHxW
            objects_per_image: list of object IDs in current frame
            queries_per_objects: count of queries on each object in current frame
            seq_objects: all objects present in the clip
        """

        splited_masks = torch.split(mask_pred, num_queries_per_object, dim=0)
        splited_queries = torch.split(queries, num_queries_per_object, dim=0)
        
        temp_out = []
        temp_que = []
        for m,q in zip(splited_masks, splited_queries):
            if len(m) >0:
                temp_out.append(torch.max(m, dim=0).values)
                temp_que.append(torch.mean(q, dim=0))

        mask_pred = torch.stack(temp_out)       # (N+1)xHxW
        queries = torch.stack(temp_que)

        # soft-aggregation
        # prob = mask_pred.clamp(1e-7, 1-1e-7)
        # logits = torch.log((prob /(1-prob)))
        # logits = F.softmax(logits, dim=0)#[1:]
        # binary = (logits > 0.5).to(torch.uint8)
        
        # binary_masks = torch.zeros((len(queries_per_object),H,W), dtype=torch.uint8)
        # c = 0
        # for i, q in enumerate(queries_per_object):
        #     if q>0:
        #         binary_masks[i][torch.where(binary[c]==1)] = 1
        #         c += 1

        # return binary_masks

        # binary to panoptic
        mask_pred = torch.argmax(mask_pred,0)
        
        # panoptic to binary - discarding overlaps
        m = []
        for obj_id in seq_objects:
            m.append((mask_pred == obj_id-1).float())
        mask_pred = torch.stack(m)
     
        return mask_pred, queries