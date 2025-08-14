import torch
import torch.nn as nn

from torch.nn import functional as F
from typing import Mapping, Tuple

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom

class Predictor:
    """
    A wrapper around DynamiteModel interactive evaluation forward pass
    """

    def __init__(self, model: nn.Module):
        self.model = model
    
    
    def get_prediction(self, inputs: Mapping) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: batched input, a dict with the following keys:
                * images: T,3,H,W tensor
                * num_clicks_per_object: TxN array
                * fg_coords_list: list of foreground clicks
                * bg_coords_list: list of background clicks
                * max_timestamp_list: timestamp of latest click in each frame
        """
        # model forward pass returns -
        # images: detectron2.ImageList wrapper around input image tensors
        # outputs: prediction logits of shape T,Q,H,W
        # num_queries_per_target: list of integers, length N+1; summing up to Q
        images, outputs, num_queries_per_target  = self.model(inputs)

        # T,N,H,W predicted binary masks and logits
        pred_masks, pred_proba = self.process_results(images,
                                                outputs,     # N
                                                num_queries_per_target)

        return pred_masks, pred_proba
    
    
    def process_results(
            self,
            images,
            mask_pred_results,
            num_queries_per_target
    ):
        """
        Args:
            outputs: dict with key "pred_masks" of shape T,Q,H,W, the prediction logits 
            num_targets: num of targets in the clip
            num_queries_per_target: count of queries on each target in each frame
            img_dims: original image dimensions
        """
        # upsample pred masks to original resolution
        mask_pred_results = F.interpolate(mask_pred_results, 
                                          size=(images.tensor.shape[-2], images.tensor.shape[-1]), 
                                          mode="bilinear", align_corners=False)

        img_dims = images.image_sizes[0]
        num_targets = len(num_queries_per_target) - 1

        processed_masks = []
        processed_logits = []
        for mask_pred_per_image in mask_pred_results:
            mask_pred_per_image = retry_if_cuda_oom(sem_seg_postprocess)(mask_pred_per_image, img_dims, img_dims[0], img_dims[1])
            mask_pred, mask_logits = retry_if_cuda_oom(self.interactive_mask_inference)(mask_pred_per_image.detach().to('cpu'), 
                                                                                    num_targets,
                                                                                    num_queries_per_target)
            processed_masks.append(mask_pred)
            processed_logits.append(mask_logits)
        return torch.stack(processed_masks), torch.stack(processed_logits)

    
    def interactive_mask_inference(
            self, 
            mask_pred, 
            num_targets,
            num_queries_per_target,
    ):
        """
        Given the raw predictions logits, obtain binary segmentation masks

        Args:
            mask_pred: raw prediction from Transformer, QxHxW
            num_targets: num of targets present in the clip
            num_queries_per_target: count of queries on each target in current frame
        """
        temp_out = []
        
        # split the mask prediction per target object, based on how many queries were
        # responsible to make prediction on that object
        splited_masks = torch.split(mask_pred, num_queries_per_target, dim=0)
        for m in splited_masks:
            if len(m) >0:
                # if there were multiple queries, take their
                # maximum response in each spatial position
                temp_out.append(torch.max(m, dim=0).values)

        mask_logits = torch.stack(temp_out)       # (N+1)xHxW

        # for each spatial position, get the channel number 
        # (== target label) where the query response is maximum
        mask_pred = torch.argmax(mask_logits,0)

        # get mask probabilities from raw prediction logits
        mask_logits = mask_logits.sigmoid()
        # for each spatial position, find the maximum probability score across all queries
        mask_proba = torch.gather(mask_logits, 0, mask_pred.unsqueeze(0)).squeeze(0)
        # for each spatial position, find which target object that maximum probability belongs to
        one_hot = torch.nn.functional.one_hot(mask_pred, num_classes=len(mask_logits))
        # only keep foreground targets
        one_hot_instances = one_hot[..., :num_targets]
        mask_proba = one_hot_instances.permute(2, 0, 1) * mask_proba

        m = []
        for obj_id in range(num_targets):
            m.append((mask_pred == obj_id).float())
        mask_pred = torch.stack(m).to(torch.uint8)
     
        return mask_pred, mask_proba