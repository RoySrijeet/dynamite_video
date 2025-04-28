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
            instances_per_frame=None,
            features=None, 
            mask_features=None, 
            multi_scale_features=None, 
            num_clicks_per_object= None,
            fg_coords = None, 
            bg_coords = None, 
            max_timestamp=None
    ):
        """
        Forward pass through the DynaMITe model

        Args:
            inputs: a list, batch output from DataLoader
            features, mask_features, multi_scale_features:
                computed once per clip and passed as an argument to avoid 
                recomputation during iterative evaluation/inference
            fg_coords: batched per clip, list of clicks coordinates for each instance in each frame
            bg_coords: batched per clip, list of background click coordinates from each frame
            num_clicks_per_object: batched per clip, number of clicks per instance in each frame
            max_timestamp: batched per clip, timestamp of last click sampled from each clip
            instances_per_frame: batched per clip, list of instances present in the frame
        """

        assert len(inputs) == 1, "Don't try more than one clip in a batch"  # TODO
        
        # extract resources from batch
        if (images is None) or (num_clicks_per_object is None) or (fg_coords is None):
            (
                images, 
                instances_per_frame, 
                num_clicks_per_object,
                fg_coords, 
                bg_coords, 
                max_timestamp
            ) = self.preprocess_batch_data(inputs)

        if features is None:
            # extract backbone features from clip frames
            features = []
            for clip_ims in images:
                clip_fs = self.backbone(clip_ims.tensor)
                features.append(clip_fs)
        
        if self.training:
            # prepare ground truth mask information
            targets = self.prepare_targets(inputs)

            for clip_idx in range(len(inputs)):
            
                clip_mask_features = mask_features[clip_idx] if mask_features is not None else None
                clip_multi_scale_features = multi_scale_features[clip_idx] if multi_scale_features is not None else None
                
                clip_outputs, clip_num_clicks_per_object, clip_num_queries_per_object = self.sem_seg_head(
                                                                                            inputs[clip_idx], 
                                                                                            images[clip_idx],
                                                                                            features[clip_idx],
                                                                                            instances_per_frame[clip_idx],
                                                                                            clip_mask_features, 
                                                                                            clip_multi_scale_features,
                                                                                            num_clicks_per_object[clip_idx],
                                                                                            fg_coords[clip_idx],
                                                                                            bg_coords[clip_idx],
                                                                                            max_timestamp[clip_idx],
                                                                                        )
                
                losses = self.criterion(clip_outputs, targets[clip_idx], clip_num_queries_per_object)

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

            (outputs, 
            mask_features, 
            multi_scale_features, 
            num_clicks_per_object, 
            num_queries_per_object) = self.sem_seg_head(
                                        inputs[0],
                                        images[0],
                                        features[0],
                                        instances_per_frame[0],
                                        mask_features, 
                                        multi_scale_features, 
                                        num_clicks_per_object[0],
                                        fg_coords[0], 
                                        bg_coords[0], 
                                        max_timestamp[0]
                                    )

            processed_results = self.process_results(images[0], outputs, instances_per_frame[0], num_queries_per_object)
            
            return (processed_results, outputs, images, instances_per_frame, features, mask_features,
                        multi_scale_features, num_clicks_per_object, fg_coords, bg_coords)


    def preprocess_batch_data(self, inputs):
        """
        Given a batch of clips, extract images as `torch.Tensor` as well as
        initial click coordinates and click count.

        Returns:
            images: list of (d2) ImageList objects, one for each clip in the batch. Each 
                ImageList object contains the image tensors of the frames in the corres
                -ponding clip as [T,3,H,W] tensors, where T: #frames in the clip
            instances_per_frame: list of instance count in each frame of each clip in the batch.
                If a clip has T frames, then one element in this list would be 
                [c1, c2, ..., cT] where cn is the #instances in the n-th frame of the clip
            num_clicks_per_object: list of click count per instance
            fg_coords: list of foreground clicks
            bg_coords: list of background clicks
        """
        images = []
        instances_per_frame = []
        num_clicks_per_object = []
        fg_coords = []
        bg_coords = []
        max_timestamp = []

        for clip in inputs:
            # convert each frame in the clip to `torch.Tensor`
            images_sample = [x.to(self.device) for x in clip["images"]]
            # normalize each frame
            images_sample = [(x - self.pixel_mean) / self.pixel_std for x in images_sample]
            # store the frames as detectron2.ImageList
            images_sample = ImageList.from_tensors(images_sample, self.size_divisibility)
            
            images.append(images_sample)
            # extract instance and click info
            instances_per_frame.append(clip["instances_per_frame"])
            num_clicks_per_object.append(clip["num_clicks_per_object"])
            fg_coords.append(clip["fg_coords_list"])
            bg_coords.append(clip["bg_coords_list"])
            max_timestamp.append(clip["max_timestamp_list"])

        return images, instances_per_frame, num_clicks_per_object, fg_coords, bg_coords, max_timestamp


    def prepare_targets(self, inputs):
        """
        Extract ground truth masks and labels of the instances. Relevant only in the training.

        Args:
            inputs: batch

        Returns:
            A list of dictionaries, one for each clip (of T frames) in the batch. Each dict contains:
                * labels - labels of the instances in the clip (a list of ints)
                * masks - binary instance masks of the frames in the clip (a list of T [N,H,W] tensors)
                * padding_mask - padding applied to the clip, [H,W] np.ndarray
                * bg_mask - background mask of the frames in the clip (a list of T [H,W] tensors)
        """

        targets = []
        for clip in inputs:
            clip_targets = []
            for i in range(clip["images"].shape[0]):
                labels = [0] * len(clip["instances_per_frame"][i])
                inst_mask = clip["instance_masks"][i].to(self.device)
                bg_mask = clip["bg_masks"][i].to(self.device)
                padding_mask = clip["padding_mask"].to(self.device)
                clip_targets.append({
                    "labels": labels,
                    "masks": inst_mask,
                    "bg_mask": bg_mask,
                    "padding_mask": padding_mask,
                })
            targets.append(clip_targets)
        return targets


    ### Evaluation ###
    def process_results(
            self, 
            images, 
            outputs, 
            instances_per_frame,
            num_queries_per_object,
    ):
        """
        Args:
            images: [T, 3, H, W] tensors of the images in the clip (d2 ImageList)
            outputs: prediction 
            instances_per_frame: list of instance IDs in the i-th frame
            num_queries_per_object: count of queries on each instance in each frame
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

        # instances in the whole clip
        seq_instances = sorted(list(set(x for ids in instances_per_frame for x in ids)))

        processed_results = []
        for mask_pred_per_image, image_size, instances_per_image, queries_per_instance in zip(mask_pred_results, images.image_sizes, instances_per_frame, num_queries_per_object):
            mask_pred_per_image = retry_if_cuda_oom(sem_seg_postprocess)(mask_pred_per_image, image_size, image_size[0], image_size[1])
            processed_r = retry_if_cuda_oom(self.interactive_instance_inference)(mask_pred_per_image, instances_per_image, queries_per_instance, seq_instances)
            processed_results.append(processed_r)

        return processed_results

    
    def interactive_instance_inference(
            self, 
            mask_pred, 
            instances_per_image, 
            queries_per_instance,
            seq_instances
    ):
        """
        Given the raw predictions from Transformer, obtain binary segmentation masks

        Args:
            mask_pred: raw prediction from Transformer, TxQxHxW
            instances_per_image: list of instance IDs in current frame
            queries_per_instances: count of queries on each instance in current frame
            seq_instances: all instances present in the clip
        """

        H,W = mask_pred.shape[1:]
        temp_out = []
        splited_masks = torch.split(mask_pred, queries_per_instance, dim=0)
        for m in splited_masks:
            if len(m) == 0:
                temp_out.append(torch.zeros(H,W).to(mask_pred.device))
            else:
                temp_out.append(torch.max(m, dim=0).values)
        
        mask_pred = torch.stack(temp_out)       # (N+1)xHxW
        mask_pred = torch.argmax(mask_pred,0)
        
        m = []
        for inst_id in seq_instances:
            if inst_id in instances_per_image:
                m.append((mask_pred == inst_id-1).float())
            else:
                m.append(torch.zeros(H,W).to(mask_pred.device))
        
        mask_pred = torch.stack(m)
     
        return mask_pred