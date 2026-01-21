import numpy as np
import torch
import torch.nn as nn

from einops import repeat
from torch import Tensor
from torch.nn import functional as F
from typing import List, Tuple


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

    def __init__(
            self,
            hidden_dim: int,
            use_no_click_query:bool=True,
            use_mlp_for_query_proj:bool=False
    ):
        
        super().__init__()
        self.hidden_dim = hidden_dim

        assert use_no_click_query or use_mlp_for_query_proj, f"Invalid learnable query initialization."
        
        self.use_no_click_query = use_no_click_query
        if self.use_no_click_query:
            # learnable query initialization for each target
            self.register_parameter("no_click_query", nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=True))
            nn.init.xavier_uniform_(self.no_click_query)

        self.use_mlp_for_query_proj = use_mlp_for_query_proj
        if self.use_mlp_for_query_proj:
            # MLP to project click description to learnable query
            self.query_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
        
    
    def forward(
            self,
            features: Tensor,
            batched_fg_coords_list: List, 
            batched_bg_coords_list: List,
            num_clicks_per_target: List,
            norms: Tuple
    ) -> Tensor:
        """
        For the given list of sampled clicks, extract queries from image features. Clicks
        follow the format [y,x, target id, frame index, click timestamp]

        Args:
            features: multi-scale feature maps of the frames in the clip. In each scale, 
                feature maps have the shape [T, C, H, W]
            batched_fg_coords_list: list of foreground clicks across all frames of the clip
            batched_bg_coords_list: list of background clicks across all frames of the clip
            norms: for normalization (height, width, last timestamp)
        
        Returns:
            descriptors: query descriptors of input clicks
            normalized_clicks_w_learnables: space-time normalized clicks corresponding to each descriptor
        """
        norm_h, norm_w, norm_t = norms
        
        # number of scales in multi-scale features
        feature_levels = len(features)
        device = features[0][0].device
        T,_,h,w = features[-1].shape
        N = len(num_clicks_per_target[0])   # num of target objects
        H = float(h*8)
        W = float(w*8)
        
        descriptors = [[] for _ in range(T)]
        normalized_clicks = [[] for _ in range(T)]
        num_queries_per_target = np.zeros(N+1, dtype=int) # including bg

        # obtain descriptor for the clicks, one click at a time, across all frames
        for fg_coords in batched_fg_coords_list:
            y,x,obj_id,fr_idx,t = fg_coords

            # extract query descriptor
            clicks = torch.tensor([fg_coords], dtype=torch.float, device=device)
            clicks = clicks[:,:2]   # [1,2]
            clicks[:,0]/=H
            clicks[:,1]/=W
            # invert (y,x) -> (x,y)
            clicks = clicks.flip(-1)
            
            click_queries = []
            # extract click features in each scale of multi-res features
            for i in range(feature_levels):
                # feature maps at i-th feature scale
                fmap_scale = features[i]
                # map of particular frame at i-th feature scale
                fmap_scale_fr = fmap_scale[fr_idx].unsqueeze(0)

                nbd_features = self.get_features_descriptors(fmap_scale_fr, clicks.unsqueeze(0))    # 1,1,D
                click_queries.append(nbd_features)

            # take the average of the features from multiple scales as the click query
            avg_click_query = torch.mean(torch.stack(click_queries, -1), dim = -1)  # 1,1,D

            # add query to frames
            for fr in range(T):
                if fr == fr_idx:
                    descriptors[fr].append(avg_click_query)
                else:
                    if self.use_no_click_query:
                        if self.use_mlp_for_query_proj:
                            learnable_query = repeat(self.no_click_query, "1 D -> 1 1 D") + self.query_proj(avg_click_query)
                        else:
                            learnable_query = repeat(self.no_click_query, "1 D -> 1 1 D")
                    else:
                        learnable_query = self.query_proj(avg_click_query)
                    descriptors[fr].append(learnable_query)
                normalized_clicks[fr].append(torch.tensor([y/norm_h, x/norm_w, obj_id/N, fr/T, t/norm_t]))
            num_queries_per_target[obj_id-1] += 1
        
        # background queries
        for bg_coords in batched_bg_coords_list:
            y,x,obj_id,fr_idx,t = bg_coords
            assert obj_id == -1
        
            # extract and scale spatial coordinates
            clicks = torch.tensor([bg_coords], dtype=torch.float, device=device)
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
            
            for fr in range(T):
                if fr == fr_idx:
                    descriptors[fr].append(avg_bg_query)
                else:
                    if self.use_no_click_query:
                        if self.use_mlp_for_query_proj:
                            learnable_query = repeat(self.no_click_query, "1 D -> 1 1 D") + self.query_proj(avg_bg_query)
                        else:
                            learnable_query = repeat(self.no_click_query, "1 D -> 1 1 D")
                    else:
                        learnable_query = self.query_proj(avg_click_query)
                    descriptors[fr].append(avg_bg_query)
                normalized_clicks[fr].append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))
            num_queries_per_target[-1] += 1

        # at this point, in each frame, there is at least one query for 
        # each target present in that frame. Each click query in one frame 
        # has corresponding learnable queries in the other frames
        descriptors = [torch.cat(desc, dim=1) for desc in descriptors]
        descriptors = torch.cat(descriptors, dim=0)
        
        normalized_clicks = [torch.stack(clks).unsqueeze(0) for clks in normalized_clicks]
        normalized_clicks = torch.cat(normalized_clicks, dim=0).to(device)
        
        return descriptors, normalized_clicks, num_queries_per_target.tolist()

    
    def get_features_descriptors(self, fmap, point_coords_per_image):

        # fmap: 1xCxHxW 
        # point_coords_per_image: 1XQx2   

        y = point_sample(fmap, point_coords_per_image, align_corners=False) # 1xCxPoints
        
        return torch.permute(y, (0, 2, 1))