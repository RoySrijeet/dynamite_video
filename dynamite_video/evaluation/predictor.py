import torch
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
            num_static_bg_queries,
    ):
        self.model = model
        self.num_overlapping_frames = num_overlapping_frames
        self.num_static_bg_queries = num_static_bg_queries
    
    
    def get_prediction(self, inputs, indices):
        """
        Args:
            inputs: batched input, a dict with the following keys:
                * images: T,3,H,W tensor
                * num_clicks_per_object: TxN array
                * fg_coords_list: list of foreground clicks
                * bg_coords_list: list of background clicks
                * max_timestamp_list: timestamp of latest click in each frame
                * query_init: dict {
                            "queries": queries of overlapping frames from prev clip
                            "clicks": clicks on overlapping frames from the prev clip
                        }

        """
        # model forward pass
        outputs, num_queries_per_target, queries, normalized_clicks  = self.model(inputs)
        # returns -
        # outputs: dict with key "pred_masks" of shape T,Q,H,W
        # num_queries_per_target: list of integers, length N+1; summing up to Q
        # queries: queries of shape T,Q,D
        # normalized_clicks: clicks of shape T,Q,5 corresponding to the queries

        img_dims = inputs[0]["images"].shape[-2:]
        processed_results = self.process_results(outputs,
                                                len(inputs[0]["num_clicks_per_object"][0]),     # N
                                                num_queries_per_target,
                                                img_dims)
            
        # T,N,H,W binary prediction masks
        pred_masks = torch.stack([x.to('cpu',dtype=torch.uint8) for x in processed_results])

        # # prepare the queries of the overlapping frames
        # # TODO - fix the forward and backward propagation case
        # overlapping_queries = queries[-self.num_overlapping_frames:].to('cpu')             # T',Q,D
        # overlapping_clicks = normalized_clicks[-self.num_overlapping_frames:].to('cpu')    # T',Q,5

        # separate the queries based on fg, bg and static_bg
        splits = [
            sum(num_queries_per_target)-num_queries_per_target[-1],     # fg queries
            num_queries_per_target[-1]-self.num_static_bg_queries,      # bg queries
            self.num_static_bg_queries                                  # static bg queries
        ]
        split_overlapping_queries = torch.split(queries, splits, dim=1)
        split_overlapping_clicks = torch.split(normalized_clicks[0], splits, dim=0)

        query_init = {
            "queries": split_overlapping_queries,
            "clicks": split_overlapping_clicks,
            "objects": normalized_clicks[:,:,2][0][:splits[0]].unique().to(torch.int).tolist()  # only fg
        }
        return pred_masks, query_init
    
    
    def process_results(
            self,
            outputs,
            num_targets,
            num_queries_per_target,
            img_dims
    ):
        """
        Args:
            outputs: dict with key "pred_masks" of shape T,Q,H,W, the prediction logits 
            num_targets: num of targets in the clip
            num_queries_per_target: count of queries on each target in each frame
            img_dims: original image dimensions
        """
        
        mask_pred_results = outputs["pred_masks"]   # [T,Q,H,W]
        # upsample masks to original resolution
        mask_pred_results = F.interpolate(mask_pred_results, size=img_dims, mode="bilinear", align_corners=False)
        del outputs

        processed_results = []
        for mask_pred_per_image in mask_pred_results:
            mask_pred_per_image = retry_if_cuda_oom(sem_seg_postprocess)(mask_pred_per_image, img_dims, img_dims[0], img_dims[1])
            processed_r = retry_if_cuda_oom(self.interactive_mask_inference)(mask_pred_per_image, 
                                                                               num_targets,
                                                                               num_queries_per_target)
            processed_results.append(processed_r)
        return processed_results

    
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

        mask_pred = torch.stack(temp_out)       # (N+1)xHxW
        mask_pred = torch.argmax(mask_pred,0)
        
        m = []
        for obj_id in range(num_targets):
            m.append((mask_pred == obj_id).float())
        mask_pred = torch.stack(m)
     
        return mask_pred