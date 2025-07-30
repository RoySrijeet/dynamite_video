import torch

from collections import defaultdict
from torch.nn import functional as F

from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom

class Predictor:
    """
    A wrapper around DynamiteModel interactive evaluation forward pass
    """


    def __init__(
            self,
            model, 
            num_overlapping_frames,
            sequence_length,
    ):
        self.model = model
        self.num_overlapping_frames = num_overlapping_frames
        self.sequence_length = sequence_length
    
    
    def get_prediction(self, inputs, indices):
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

        # T,N,H,W predictions masks
        pred_masks, pred_proba = self.process_results(images,
                                                outputs,     # N
                                                num_queries_per_target)

        if self.num_overlapping_frames == 0:
            return pred_masks, None
        ## jerry-built. TODO
        if self.sequence_length - 1 in indices:
            return pred_masks, None
        
        # overlapping frames
        overlapping_pred_masks = pred_masks[-self.num_overlapping_frames:]
        overlapping_pred_proba = pred_proba[-self.num_overlapping_frames:]
        
        # find overlapping targets from binary masks
        overlapping_targets = overlapping_pred_masks.any(dim=(0, 2, 3)).nonzero(as_tuple=True)[0]
        
        # find frames where each target shows max confidence
        max_response_frames = overlapping_pred_proba.amax(dim=(2, 3)).argmax(dim=0)
        # keep only the predicted targets
        max_response_frames = max_response_frames[overlapping_targets]
        
        # for each predicted target, record high confidence areas
        overlapping_pred_proba = overlapping_pred_proba > 0.9
        
        frames_to_sample = defaultdict(list)
        overlapping_masks = defaultdict(list)
        for fr_idx, tgt_id in zip(max_response_frames, overlapping_targets):
            frames_to_sample[indices[self.num_overlapping_frames + fr_idx.item()]].append(tgt_id.item())
            overlapping_masks[indices[self.num_overlapping_frames + fr_idx.item()]].append(overlapping_pred_proba[fr_idx][tgt_id])

        overlap = {
            "frames": frames_to_sample,
            "masks": overlapping_masks
        }
        return pred_masks, overlap
    
    
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
        splited_masks = torch.split(mask_pred, num_queries_per_target, dim=0)
        for m in splited_masks:
            if len(m) >0:
                temp_out.append(torch.max(m, dim=0).values)

        mask_logits = torch.stack(temp_out)       # (N+1)xHxW
        mask_pred = torch.argmax(mask_logits,0)
        
        m = []
        for obj_id in range(num_targets):
            m.append((mask_pred == obj_id).float())
        mask_pred = torch.stack(m).to(torch.uint8)
     
        return mask_pred, mask_logits[:-1].sigmoid()  # discard bg