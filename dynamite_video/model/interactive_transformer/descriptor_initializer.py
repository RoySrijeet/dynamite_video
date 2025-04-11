import copy
import torch
import torch.nn as nn

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
        self.no_click_query = nn.Embedding(1, hidden_dim)
        
        self.register_parameter("bg_query", nn.Parameter(torch.zeros(hidden_dim), False))

    
    # QUERY NOT STACKING - GROUP FRAME WISE (TxQxD)
    def forward(
            self,
            features: Tensor,
            batched_fg_coords_list: List, 
            batched_bg_coords_list: List,
            img_dims: Tuple,
            max_timestamp: List,
            use_static_bg_queries: bool,
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
        
        
        # stack queries for each frame
        for fr_idx, fr_fg_coords in enumerate(batched_fg_coords_list):
            
            # foreground queries of current frame
            fr_fg_queries = []
            fr_fg_normalized_clicks = []
            
            # stack queries for each instance in the frame
            for inst_id, inst_fg_coords in enumerate(fr_fg_coords):
                # if there are no clicks on a certain instance, insert empty query
                if len(inst_fg_coords) ==0:
                    fr_fg_queries.append(self.no_click_query.weight.unsqueeze(1).to(device))
                    fr_fg_normalized_clicks.append(torch.tensor([-1.0, -1.0, -1.0]))
                    continue

                for coords in inst_fg_coords:
                    fr_fg_normalized_clicks.append(torch.tensor([coords[0]/norm_h, coords[1]/norm_w, coords[-1]/norm_t]))

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

        descriptors = [torch.cat(desc, dim=1) for desc in descriptors]
        # pad descriptors of each frame so that they all have same length
        max_queries = max([desc.shape[1] for desc in descriptors])
        for i, desc in enumerate(descriptors):
            if use_static_bg_queries:
                pad = max_queries-desc.shape[1]
                bg_queries = repeat(self.bg_query, "C -> 1 L C", L=pad)
            else:
                pad = max_queries+1-desc.shape[1]
                bg_queries = repeat(self.bg_query, "C -> 1 L C", L=pad)
            descriptors[i] = torch.cat((descriptors[i], bg_queries), dim=1)
        descriptors = torch.cat(descriptors, dim=0)  # TxQxD
        
        for idx in range(len(normalized_clicks)):
            clks = normalized_clicks[idx]
            if len(clks) < max_queries:
                normalized_clicks[idx].extend([torch.tensor([-1.0, -1.0, -1.0])] * (max_queries-len(clks)))

        normalized_clicks = [torch.stack(clks).unsqueeze(0) for clks in normalized_clicks]
        normalized_clicks = torch.cat(normalized_clicks, dim=0).to(device)
        
        return descriptors, normalized_clicks

    # QUERY STACKING - GROUPED INSTANCE-WISE, THEN STACKED
    # def forward(
    #         self,
    #         features: Tensor,
    #         batched_fg_coords_list: List, 
    #         batched_bg_coords_list: List,
    #         img_dims: Tuple,
    #         max_timestamp: List,
    # ) -> Tensor:
    #     """
    #     For the given list of sampled clicks, extract queries from image features

    #     Queries corresponding to clicks sampled on different frames are stacked together to generate a single 
    #     QxD tensor, where Q = number of foreground and background clicks + padding clicks. Padding clicks are
    #     intended to be placeholder queries for frames where an instance didn't receive a click. The idea is,
    #     that, each instance in each frame gets at least one query.

    #     Args:
    #         features: multi-scale feature maps of the frames in the clip. In each scale, 
    #             feature maps have the shape [T, C, H, W]
    #         batched_fg_coords_list: list of foreground clicks across all frames of the clip
    #         batched_bg_coords_list: list of background clicks across all frames of the clip
    #         img_dims: image (height, width)
    #         max_timestamp: latest timestamp on each frame
        
    #     Returns:
    #         descriptors: query descriptors of input clicks
    #         normalized_clicks_coords: space-time normalized clicks corresponding to each descriptor
    #     """
    #     norm_t = max(max_timestamp)
    #     norm_h, norm_w = img_dims
        
    #     # number of feature scales
    #     feature_levels = len(features)
    #     device = features[0][0].device
    #     _,d,h,w = features[-1].shape
    #     H = float(h*8)
    #     W = float(w*8)
        
    #     max_num_instances = len(batched_fg_coords_list[0])
    #     descriptors = [[] for _ in range(max_num_instances)]
    #     normalized_clicks = [[] for _ in range(max_num_instances)]
        
    #     for fr_idx, fr_fg_coords in enumerate(batched_fg_coords_list):

    #         # separate the clicks into bins based on the instance it was sampled from
    #         for inst_id, inst_fg_coords in enumerate(fr_fg_coords):

    #             if len(inst_fg_coords)==0:
    #                 descriptors[inst_id].append(self.no_click_query.weight.unsqueeze(1).to(device))
    #                 normalized_clicks[inst_id].append(torch.tensor([-1.0, -1.0, -1.0]))
    #                 continue
                
    #             for coords in inst_fg_coords:
    #                 normalized_clicks[inst_id].append(torch.tensor([coords[0]/norm_h, coords[1]/norm_w, coords[-1]/norm_t]))

    #             clicks = torch.tensor(inst_fg_coords, dtype=torch.float, device=device)
    #             # extract and scale spatial coordinates
    #             clicks = clicks[:,:2]
    #             clicks[:,0]/=H
    #             clicks[:,1]/=W
    #             # invert (y,x) -> (x,y)
    #             clicks = clicks.flip(-1)

    #             inst_queries = []
    #             # extract click features in each scale of multi-res features
    #             for i in range(feature_levels):
    #                 # feature maps at i-th feature scale
    #                 fmap_scale = features[i]
    #                 # map of particular frame at i-th feature scale
    #                 fmap_scale_fr = fmap_scale[fr_idx].unsqueeze(0)

    #                 nbd_features = self.get_features_descriptors(fmap_scale_fr, clicks.unsqueeze(0))
    #                 inst_queries.append(nbd_features)

    #             # take the average of the features from multiple scales as the query for the clicks on this instance
    #             avg_inst_query = torch.mean(torch.stack(inst_queries, -1), dim = -1)
    #             descriptors[inst_id].append(avg_inst_query)
        
        
    #     # background queries
    #     # continue stacking frame-wise
    #     bg_descriptors = []
    #     bg_normalized_clicks = []
    #     for fr_idx, fr_bg_coords in enumerate(batched_bg_coords_list):
            
    #         if len(fr_bg_coords) == 0:
    #             continue
            
    #         for coords in fr_bg_coords:
    #             bg_normalized_clicks.append(torch.tensor([coords[0]/norm_h, coords[1]/norm_w, coords[-1]/norm_t]))
            
    #         clicks = torch.tensor(fr_bg_coords, dtype=torch.float, device=device)
    #         # extract and scale spatial coordinates
    #         clicks = clicks[:,:2]
    #         clicks[:,0]/=H
    #         clicks[:,1]/=W
    #         # invert (y,x) -> (x,y)
    #         clicks = clicks.flip(-1)

    #         bg_queries = []
    #         for i in range(feature_levels):
    #             # maps at i-th feature scale
    #             fmap_scale = features[i]
    #             # map of particular frame at i-th feature scale
    #             fmap_scale_fr = fmap_scale[fr_idx].unsqueeze(0)

    #             nbd_features = self.get_features_descriptors(fmap_scale_fr, clicks.unsqueeze(0))
    #             bg_queries.append(nbd_features)
            
    #         avg_bg_query = torch.mean(torch.stack(bg_queries, -1), dim = -1)
    #         bg_descriptors.append(avg_bg_query)
        
    #     if len(bg_descriptors) > 0:
    #         descriptors.append(bg_descriptors)
    #         normalized_clicks.append(bg_normalized_clicks)
        
    #     descriptors = torch.cat([torch.cat(desc, dim=1) for desc in descriptors], dim=1).squeeze(0)
    #     normalized_clicks = torch.cat([torch.stack(clcks) for clcks in normalized_clicks], dim=0).to(device)
        
    #     return descriptors , normalized_clicks
    
    
    # QUERY STACKING - GROUPED FRAME-WISE, THEN STACKED
    # def forward(
    #         self, 
    #         features: Tensor, 
    #         batched_fg_coords_list: List, 
    #         batched_bg_coords_list: List,
    #         img_dims: Tuple,
    #         max_timestamp:List,
    # ) -> Tensor:
    #     """
    #     For the given list of sampled clicks, extract queries from image features

    #     The logic behind stacking the queries is as follows -

    #     If frame i has n_i foreground clicks and b_i background clicks. Then, the query
    #     vector, Q, has (n_0 + n_1 + ... + b_0 + b_1 + ...) query components where
    #     Q[:n_0] corresponds to the foreground clicks on frame 0,
    #     Q[n_0+1:n_0+n_1] corresponds to the foreground clicks on frame 1, 
    #     and so on for the n_N total foreground clicks and then,
    #     Q[n_N+1:n_N+b_0] corresponds to the background clicks on frame 0,
    #     and so on stacking all the background clicks.

    #     Note, the foreground queries in each frame also follow the same order as the
    #     clicks on the instances. So, if Q[:n_0] are the foreground queries of the first
    #     frame, then Q[:i_0] are the queries corresponding to the clicks on the first
    #     instance of the frame, Q[i_0+1:i_0+i_1] are the queries corresponding to the clicks
    #     on the second instances of the frame and so on (i_0 + i_1 + ... + i_p = n_0 where 
    #     i_j is the click count on the j-th instance in frame 0).

    #     If an instance in a frame has no foreground clicks on it, insert an empty learnable
    #     query in its stead.
        
    #     Args:
    #         features: multi-scale feature maps of the frames in the clip. In each scale, 
    #             feature maps have the shape [T, C, H, W]
    #         batched_fg_coords_list: list of foreground clicks across all frames of the clip
    #         batched_bg_coords_list: list of background clicks across all frames of the clip
    #         img_dims: image (height, width)
    #         max_timestamp: latest timestamp on each frame
        
    #     Returns:
    #         descriptors: query descriptors of input clicks
    #         normalized_clicks_coords: space-time normalized clicks corresponding to each descriptor
    #     """
    #     norm_t = max(max_timestamp)
    #     norm_h, norm_w = img_dims
        
    #     # number of feature scales
    #     feature_levels = len(features)
    #     device = features[0][0].device
    #     _,d,h,w = features[-1].shape
    #     H = float(h*8)
    #     W = float(w*8)
        
    #     descriptors = []
    #     normalized_clicks = []
        
    #     # foreground queries
    #     fg_queries = []
    #     # stack queries for each frame
    #     for fr_idx, fr_fg_coords in enumerate(batched_fg_coords_list):
            
    #         # stack queries for each instance in the frame
    #         for inst_id, inst_fg_coords in enumerate(fr_fg_coords):
    #             # if there are no clicks on a certain instance, insert empty query
    #             if len(inst_fg_coords) ==0:
    #                 fg_queries.append(self.no_click_query.weight.unsqueeze(1).to(device))
    #                 normalized_clicks.append(torch.tensor([-1.0, -1.0, -1.0]))
    #                 continue

    #             for coords in inst_fg_coords:
    #                 normalized_clicks.append(torch.tensor([coords[0]/norm_h, coords[1]/norm_w, coords[-1]/norm_t]))

    #             clicks = torch.tensor(inst_fg_coords, dtype=torch.float, device=device)
    #             # extract and scale spatial coordinates
    #             clicks = clicks[:,:2]
    #             clicks[:,0]/=H
    #             clicks[:,1]/=W
    #             # invert (y,x) -> (x,y)
    #             clicks = clicks.flip(-1)
                
    #             inst_queries = []
    #             # extract click features in each scale of multi-res features
    #             for i in range(feature_levels):
    #                 # feature maps at i-th feature scale
    #                 fmap_scale = features[i]
    #                 # map of particular frame at i-th feature scale
    #                 fmap_scale_fr = fmap_scale[fr_idx].unsqueeze(0)

    #                 nbd_features = self.get_features_descriptors(fmap_scale_fr, clicks.unsqueeze(0))
    #                 inst_queries.append(nbd_features)

    #             # take the average of the features from multiple scales as the query for the clicks on this instance
    #             avg_inst_query = torch.mean(torch.stack(inst_queries, -1), dim = -1)
    #             fg_queries.append(avg_inst_query)
        
        
    #     # background queries
    #     # continue stacking frame-wise
    #     for fr_idx, fr_bg_coords in enumerate(batched_bg_coords_list):
            
    #         if len(fr_bg_coords) == 0:
    #             # fg_queries.append(self.no_click_query.weight.unsqueeze(1).to(device)) # TODO
    #             # normalized_clicks.append(torch.tensor([-1.0, -1.0, -1.0]))
    #             continue
            
    #         for coords in fr_bg_coords:
    #             normalized_clicks.append(torch.tensor([coords[0]/norm_h, coords[1]/norm_w, coords[-1]/norm_t]))
            
    #         clicks = torch.tensor(fr_bg_coords, dtype=torch.float, device=device)
    #         # extract and scale spatial coordinates
    #         clicks = clicks[:,:2]
    #         clicks[:,0]/=H
    #         clicks[:,1]/=W
    #         # invert (y,x) -> (x,y)
    #         clicks = clicks.flip(-1)

    #         bg_queries = []
    #         for i in range(feature_levels):
    #             # maps at i-th feature scale
    #             fmap_scale = features[i]
    #             # map of particular frame at i-th feature scale
    #             fmap_scale_fr = fmap_scale[fr_idx].unsqueeze(0)

    #             nbd_features = self.get_features_descriptors(fmap_scale_fr, clicks.unsqueeze(0))
    #             bg_queries.append(nbd_features)
            
    #         avg_bg_query = torch.mean(torch.stack(bg_queries, -1), dim = -1)
    #         fg_queries.append(avg_bg_query)

    #     descriptors = torch.cat(fg_queries, dim=1).squeeze(0)
    #     normalized_clicks = torch.stack(normalized_clicks).to(device)
    #     return descriptors, normalized_clicks
    

    def get_features_descriptors(self, fmap, point_coords_per_image):

        # fmap: 1xCxHxW 
        # point_coords_per_image: 1XQx2   

        y = point_sample(fmap, point_coords_per_image, align_corners=False) # 1xCxPoints
        
        return torch.permute(y, (0, 2, 1))