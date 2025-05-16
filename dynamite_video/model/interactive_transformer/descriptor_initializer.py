import copy
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from einops import repeat
from itertools import chain
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from torch.nn import functional as F

def point_sample(input, point_coords, **kwargs):
    """
    Source: https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


class AvgClicksPoolingInitializer(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.register_parameter("no_click_query", nn.Parameter(torch.zeros(hidden_dim), requires_grad=True))

    
    # QUERY NOT STACKING - GROUP FRAME WISE (TxQxD)
    def forward(
            self,
            features: Tensor,
            instances_per_frame: List,
            batched_fg_coords_list: List, 
            batched_bg_coords_list: List,
            img_dims: Tuple,
            max_timestamp: List,
    ) -> Tensor:
        """
        For the given list of sampled clicks, extract queries from image features

        Queries belonging to the same frame are grouped together. In each frame, each instance
        must have at least one query (corresponding to one click). If there is no click sampled
        on a given instance in the current frame, use a padding query

        Args:
            features: multi-scale feature maps of the frames in the clip. In each scale, 
                feature maps have the shape [T, C, H, W]
            batched_fg_coords_list: list of foreground clicks across all frames of the clip
            batched_bg_coords_list: list of background clicks across all frames of the clip
            img_dims: image (height, width)
            max_timestamp: latest timestamp on each frame
        
        Returns:
            descriptors: query descriptors of input clicks
            normalized_clicks_coords: space-time normalized clicks corresponding to each descriptor
        """

        norm_t = max(max_timestamp)
        norm_h, norm_w = img_dims
        
        # number of feature scales
        feature_levels = len(features)
        device = features[0][0].device
        _,d,h,w = features[-1].shape
        H = float(h*8)
        W = float(w*8)
        
        descriptors = []
        normalized_clicks = []
        num_queries_per_object = np.zeros((len(batched_fg_coords_list), len(batched_fg_coords_list[0])+1)).astype('int')
        
        # stack queries for each frame
        for fr_idx, fr_fg_coords in enumerate(batched_fg_coords_list):
            
            # foreground queries of current frame
            fr_fg_queries = []
            fr_fg_normalized_clicks = []
            
            # stack queries for each instance in the frame
            for inst_id, inst_fg_coords in enumerate(fr_fg_coords):

                if inst_id+1 not in instances_per_frame[fr_idx]:
                    continue

                # if there are no clicks on a certain instance, insert empty query
                if len(inst_fg_coords) == 0:
                    fr_fg_queries.append(repeat(self.no_click_query, "C -> 1 1 C"))
                    fr_fg_normalized_clicks.append(torch.tensor([-1.0, -1.0, -1.0]))
                    num_queries_per_object[fr_idx][inst_id] += 1
                    continue

                for coords in inst_fg_coords:
                    fr_fg_normalized_clicks.append(torch.tensor([coords[0]/norm_h, coords[1]/norm_w, coords[-1]/norm_t]))
                    num_queries_per_object[fr_idx][inst_id] += 1

                clicks = torch.tensor(inst_fg_coords, dtype=torch.float, device=device)
                # extract and scale spatial coordinates
                clicks = clicks[:,:2]
                clicks[:,0]/=H
                clicks[:,1]/=W
                # invert (y,x) -> (x,y)
                clicks = clicks.flip(-1)
                
                inst_queries = []
                # extract click features in each scale of multi-res features
                for i in range(feature_levels):
                    # feature maps at i-th feature scale
                    fmap_scale = features[i]
                    # map of particular frame at i-th feature scale
                    fmap_scale_fr = fmap_scale[fr_idx].unsqueeze(0)

                    nbd_features = self.get_features_descriptors(fmap_scale_fr, clicks.unsqueeze(0))
                    inst_queries.append(nbd_features)

                # take the average of the features from multiple scales as the query for the clicks on this instance
                avg_inst_query = torch.mean(torch.stack(inst_queries, -1), dim = -1)
                fr_fg_queries.append(avg_inst_query)
            
            descriptors.append(fr_fg_queries)
            normalized_clicks.append(fr_fg_normalized_clicks)
        
        # background queries
        for fr_idx, fr_bg_coords in enumerate(batched_bg_coords_list):

            if len(fr_bg_coords) == 0:
                continue
            
            for coords in fr_bg_coords:
                normalized_clicks[fr_idx].append(torch.tensor([coords[0]/norm_h, coords[1]/norm_w, coords[-1]/norm_t]))
                num_queries_per_object[fr_idx][-1] += 1
            
            clicks = torch.tensor(fr_bg_coords, dtype=torch.float, device=device)
            # extract and scale spatial coordinates
            clicks = clicks[:,:2]
            clicks[:,0]/=H
            clicks[:,1]/=W
            # invert (y,x) -> (x,y)
            clicks = clicks.flip(-1)

            fr_bg_queries = []
            for i in range(feature_levels):
                # maps at i-th feature scale
                fmap_scale = features[i]
                # map of particular frame at i-th feature scale
                fmap_scale_fr = fmap_scale[fr_idx].unsqueeze(0)

                nbd_features = self.get_features_descriptors(fmap_scale_fr, clicks.unsqueeze(0))
                fr_bg_queries.append(nbd_features)
            
            avg_bg_query = torch.mean(torch.stack(fr_bg_queries, -1), dim = -1)
            descriptors[fr_idx].append(avg_bg_query)

        # at this point, in each frame, there is at least one query for 
        # each instance present in that frame
        descriptors = [
                        torch.cat(desc, dim=1) if len(desc) > 0 else torch.zeros((1, 0, self.hidden_dim), device=device)
                        for desc in descriptors
                    ]
        
        return descriptors, normalized_clicks, num_queries_per_object.tolist()
    

    def get_features_descriptors(self, fmap, point_coords_per_image):

        # fmap: 1xCxHxW 
        # point_coords_per_image: 1XQx2   

        y = point_sample(fmap, point_coords_per_image, align_corners=False) # 1xCxPoints
        
        return torch.permute(y, (0, 2, 1))