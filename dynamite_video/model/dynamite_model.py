# Adapted from https://github.com/amitrana001/DynaMITe


import os
import copy
import torch
import pickle

from torch import nn
from torch.nn import functional as F
from typing import Tuple

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

from dynamite_video.model.utils.criterion import SetFinalCriterion


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
        debug: bool,
        save_dir: str,
        
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
        
        # debug
        self.debug = debug
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        if self.debug:
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)

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
            
            # debug
            "debug": cfg.DEBUG,
            "save_dir": os.path.join(cfg.OUTPUT_DIR, "debug"),
        }
        
    
    @property
    def device(self):
        return self.pixel_mean.device

    
    def forward(self, inputs, images=None,  num_instances=None,
                features=None, mask_features=None, 
                multi_scale_features=None, num_clicks_per_object= None,
                fg_coords = None, bg_coords = None, max_timestamp=None):
        """
        Forward pass through the DynaMITe model

        Args:
            inputs: a list, batch output from DataLoader.
                    Each item in the list, is a dictionary and contains inputs for 
                    one sample - a clip. The dictionary contains following keys:
                * images: [T,3,H,W] np.ndarray, RGB images of the clip frames
                * instance_masks: [T,N,H,W] np.ndarray, binary segmentation masks 
                    of the instances in each frame of the clip
                * semantic_masks: [T,H,W] np.ndarray, semantic map of each frame
                * bg_masks: [T,H,W] np.ndarray, background mask of each frame
                * padding_mask: [H,W] np.ndarray, padding applied
                * instance_ids: list(int), IDs of the instances present in the cliip
                * num_instances_per_frame: list(int), num of instances present in each
                    frame of the clip
                * frame_instance_occupancy: dict, mapping between instance ID and 
                    frame indices where that instance appears in the clip
                * fg_coords_list: list, foreground clicks sampled on each frame (at an
                    instance-level)
                * bg_coordsl_list: list, background clicks samples on each frame
                * num_clicks_per_object: list, num of foreground sampled on each instance,
                    in each frame
                * max_timestamp_list: list, timestamp of last click sampled from each clip
                * meta: metadata of the source sequence (original resolution, sequence name, 
                    mappings between original and serialized/model instance IDs)
            features, mask_features, multi_scale_features:
                computed once per clip and passed as an argument to avoid recomputation
                during iterative evaluation/inference
            fg_coords: a batched list where each item is
                * list of list of clicks coordinates for each instance in each frame
            bg_coords: a batched list where each item is
                * list of background click coordinates from each frame
            num_clicks_per_object: a batched list where each item  is
                * list of number of clicks per instance in each frame
            max_timestamp: a batched list where each item is
                * timestamp of last click sampled from each clip
            num_instances:  a batched list where each item is
                * a list of num of instances in frame
        Returns:
            list[Instances]:
                each Instances has the predicted masks for one image.

        """
        # NOTE: Currently batch size is fixed to 1.
        assert len(inputs) == 1, "Don't try more than one clip in a batch"
        
        # extract resources from batch
        if (images is None) or (num_clicks_per_object is None) or (fg_coords is None):
            (
                images, 
                num_instances, 
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

        sample_name = inputs[0]["meta"]["seq_name"] + "_".join([str(idx) for idx in inputs[0]["meta"]["frame_indices"]]) # debug
        sample_name = sample_name.replace('/', '-')
        self.sample_save_dir = os.path.join(self.save_dir, sample_name)
        os.makedirs(self.sample_save_dir, exist_ok=True)
        if self.debug:
            sample_name = inputs[0]["meta"]["seq_name"] + "_".join([str(idx) for idx in inputs[0]["meta"]["frame_indices"]]) # debug
            sample_name = sample_name.replace('/', '-')
            self.sample_save_dir = os.path.join(self.save_dir, sample_name)
            os.makedirs(self.sample_save_dir, exist_ok=True)
            with open(os.path.join(self.sample_save_dir, f"batch.pkl"), "wb") as f:
                pickle.dump(inputs, f)
            torch.save(features, os.path.join(self.sample_save_dir, f"backbone_image_features.pth"))
        
        if self.training:
            # prepare ground truth mask information
            targets = self.prepare_targets(inputs)

            for sample_idx in range(len(inputs)):
            
                sample_mask_features = mask_features[sample_idx] if mask_features is not None else None
                sample_multi_scale_features = multi_scale_features[sample_idx] if multi_scale_features is not None else None
                
                sample_outputs, sample_num_clicks_per_object = self.sem_seg_head(
                                                                        inputs[sample_idx], 
                                                                        images[sample_idx],
                                                                        features[sample_idx],
                                                                        num_instances[sample_idx],
                                                                        sample_mask_features, 
                                                                        sample_multi_scale_features,
                                                                        num_clicks_per_object[sample_idx],
                                                                        fg_coords[sample_idx],
                                                                        bg_coords[sample_idx],
                                                                        max_timestamp[sample_idx],
                                                                    )
                
                losses = self.criterion(sample_outputs, targets[sample_idx], sample_num_clicks_per_object)

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

            (outputs, mask_features, multi_scale_features, num_clicks_per_object) = self.sem_seg_head(
                                                                                            inputs[0],
                                                                                            images[0],
                                                                                            features[0],
                                                                                            num_instances[0],
                                                                                            mask_features, 
                                                                                            multi_scale_features, 
                                                                                            num_clicks_per_object[0],
                                                                                            fg_coords[0], 
                                                                                            bg_coords[0], 
                                                                                            max_timestamp[0])
            # if self.debug:
            torch.save(outputs["pred_masks"], os.path.join(self.sample_save_dir, f"raw_predictions.pth"))
            processed_results = self.process_results(inputs[0], images[0], outputs, num_instances[0], num_clicks_per_object)
            if self.iterative_evaluation:
                return (processed_results, outputs, images,  num_instances, features, mask_features,
                        multi_scale_features, num_clicks_per_object, fg_coords, bg_coords)
            else:
                return processed_results


    def preprocess_batch_data(self, inputs):
        """
        Given a batch of clips, extract images as `torch.Tensor` as well as
        initial click coordinates and click count.

        Returns:
            images: list of (d2) ImageList objects, one for each clip in the batch. Each 
                ImageList object contains the image tensors of the frames in the corres
                -ponding clip as [T,3,H,W] tensors, where T: #frames in the clip
            num_instances: list of instance count in each frame of each clip in the batch.
                If a clip has T frames, then one element in this list would be 
                [c1, c2, ..., cT] where cn is the #instances in the n-th frame of the clip
            num_clicks_per_object: list of click count per instance
            fg_coords: list of foreground clicks
            bg_coords: list of background clicks
        """
        images = []
        num_instances = []
        num_clicks_per_object = []
        fg_coords = []
        bg_coords = []
        max_timestamp = []

        for clip in inputs:
            # convert each frame in the clip to `torch.Tensor`
            images_sample = [torch.from_numpy(x).to(self.device) for x in clip["images"]]
            # normalize each frame
            images_sample = [(x - self.pixel_mean) / self.pixel_std for x in images_sample]
            # store the frames as detectron2.ImageList
            images_sample = ImageList.from_tensors(images_sample, self.size_divisibility)
            
            images.append(images_sample)
            # extract instance and click info
            num_instances.append(clip["num_instances_per_frame"])
            num_clicks_per_object.append(clip["num_clicks_per_object"])
            fg_coords.append(clip["fg_coords_list"])
            bg_coords.append(clip["bg_coords_list"])
            max_timestamp.append(clip["max_timestamp_list"])

        return images, num_instances, num_clicks_per_object, fg_coords, bg_coords, max_timestamp


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
                labels = [0] * clip["num_instances_per_frame"][i]
                inst_mask = torch.from_numpy(clip["instance_masks"][i]).to(self.device)
                bg_mask = torch.from_numpy(clip["bg_masks"][i]).to(self.device)
                padding_mask = torch.from_numpy(clip["padding_mask"]).to(self.device)
                clip_targets.append({
                    "labels": labels,
                    "masks": inst_mask,
                    "bg_mask": bg_mask,
                    "padding_mask": padding_mask,
                })
            targets.append(clip_targets)
        return targets


    def process_results(self, inputs, images, outputs, num_instances, num_clicks_per_object):
        """TODO for EVAL - Query Stacking
        Process results after one forward pass through the iterative evaluation

        Args:
            inputs: current clip
            images: d2 ImageList, [T, 3, H, W] tensors of the images in the clip
            outputs: prediction 
            num_instances: List [n_1, n_2, ..., n_T] where n_i is the #instances in the i-th frame
            num_clicks_per_object: count of clicks on each instance in each frame
        """
       
        mask_pred_results = outputs["pred_masks"]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del outputs

        processed_results = []
        
        for mask_pred_per_image, image_size, inst_per_image, clicks_per_image in zip(mask_pred_results, images.image_sizes, num_instances, num_clicks_per_object):
            height, width = image_size[0], image_size[1]

            mask_pred_per_image = retry_if_cuda_oom(sem_seg_postprocess)(mask_pred_per_image, image_size, height, width)

            # interactive instance segmentation inference
            instance_r = retry_if_cuda_oom(self.interactive_instance_inference)(mask_pred_per_image, inst_per_image, clicks_per_image)
            processed_results.append(instance_r)

        processed_results = torch.stack(processed_results)
        return processed_results
    
    
    def interactive_instance_inference(self, mask_pred, num_instances, num_clicks_per_object):
        """TODO for EVAL - Query stacking"""

        num_clicks_per_object_copy = copy.deepcopy(num_clicks_per_object)
        # handle zero clicks 
        for i in range(len(num_clicks_per_object_copy)):
            if num_clicks_per_object_copy[i] == 0:
                num_clicks_per_object_copy[i]+=1
        num_clicks_per_object_copy.append(mask_pred.shape[0]-sum(num_clicks_per_object_copy))
        
        temp_out = []
        if num_clicks_per_object_copy[-1] == 0:
            splited_masks = torch.split(mask_pred, num_clicks_per_object_copy[:-1], dim=0)
        else:
            splited_masks = torch.split(mask_pred, num_clicks_per_object_copy, dim=0)
        for m in splited_masks:
            temp_out.append(torch.max(m, dim=0).values)
        
        mask_pred = torch.stack(temp_out)
        mask_pred = torch.argmax(mask_pred,0)
        
        if num_instances > 0:
            # if num_instances > 25:
            #     raise 
            m = []
            for i in range(num_instances):
                m.append((mask_pred == i).float())
            
            mask_pred = torch.stack(m)
        else:
            assert mask_pred.ndim == 2
     
        return mask_pred