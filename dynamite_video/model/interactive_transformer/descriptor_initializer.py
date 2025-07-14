import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from einops import repeat
from torch import Tensor
from torch.nn import functional as F
from typing import Dict, List, Tuple


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

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # learnable query for each target
        self.register_parameter("no_click_query", nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=True))
        
        nn.init.xavier_uniform_(self.no_click_query)
        
    
    def forward(
            self,
            features: Tensor,
            batched_fg_coords_list: List, 
            batched_bg_coords_list: List,
            num_clicks_per_target: List,
            norms: Tuple,
            query_init: None|Dict=None,
    ) -> Tensor:
        """
        For the given list of sampled clicks, extract queries from image features

        Queries belonging to the same frame are grouped together. In each frame, each target
        must have at least one query (corresponding to one click).

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
        if query_init is not None:
            # evaluation path
            return self.forward_eval(
                features,
                batched_fg_coords_list,
                batched_bg_coords_list,
                num_clicks_per_target,
                norms,
                query_init
            )

        norm_h, norm_w, norm_t = norms
        
        # number of feature scales
        feature_levels = len(features)
        device = features[0][0].device
        T,_,h,w = features[-1].shape
        N = len(num_clicks_per_target[0]) + 1 # add bg
        H = float(h*8)
        W = float(w*8)
        
        descriptors = [[] for _ in range(T)]
        normalized_clicks = [[] for _ in range(T)]
        num_queries_per_target = [0 for _ in range(N)]

        # obtain queries one click at a time, across all frames
        for fg_coords in batched_fg_coords_list:
            y,x,obj_id,fr_idx,t = fg_coords

            for fr, desc, clks in zip(range(T), descriptors, normalized_clicks):
                if fr==fr_idx:
                    # normalize the click
                    clks.append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))
                    num_queries_per_target[obj_id-1] += 1

                    # extract query descriptor
                    clicks = torch.tensor([fg_coords], dtype=torch.float, device=device)
                    clicks = clicks[:,:2]
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
                    avg_click_query = torch.mean(torch.stack(click_queries, -1), dim = -1)
                    desc.extend(torch.split(avg_click_query, 1, dim=1))
                else:
                    desc.append(repeat(self.no_click_query, "1 C -> 1 1 C"))
                    clks.append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))
        
        # background queries
        for bg_coords in batched_bg_coords_list:
            y,x,obj_id,fr_idx,t = bg_coords
            assert obj_id == -1
        
            for fr, desc, clks in zip(range(T), descriptors, normalized_clicks):
                if fr==fr_idx:
                    clks.append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))
                    num_queries_per_target[-1] += 1
        
                    clicks = torch.tensor([bg_coords], dtype=torch.float, device=device)
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
                    desc.extend(torch.split(avg_bg_query, 1, dim=1))
                else:
                    desc.append(repeat(self.no_click_query, "1 C -> 1 1 C"))
                    clks.append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))

        # at this point, in each frame, there is at least one query for 
        # each target present in that frame. Each click query in one frame 
        # has corresponding learnable queries in the other frames
        descriptors = [torch.cat(desc, dim=1) for desc in descriptors]
        descriptors = torch.cat(descriptors, dim=0)
        
        normalized_clicks = [torch.stack(clks).unsqueeze(0) for clks in normalized_clicks]
        normalized_clicks = torch.cat(normalized_clicks, dim=0).to(device)
        
        return descriptors, normalized_clicks, num_queries_per_target
    

    def forward_eval(
            self,
            features: Tensor,
            batched_fg_coords_list: List, 
            batched_bg_coords_list: List,
            num_clicks_per_target: List,
            norms: Tuple,
            query_init: Dict,
    ) -> Tensor:
        """
        During evaluation, the overlapping queries and clicks must be carefully handled.
        The overlapping queries and new click queries are stacked serially with foreground
        queries, background queries and static bg queries.
        """
        norm_h, norm_w, norm_t = norms
        
        # number of feature scales
        feature_levels = len(features)
        device = features[0][0].device
        T,_,h,w = features[-1].shape
        N = len(num_clicks_per_target[0]) + 1 # add bg
        H = float(h*8)
        W = float(w*8)
        
        descriptors = [[] for _ in range(T)]
        normalized_clicks = [[] for _ in range(T)]

        # if there are overlapping frames, first add the overlapping queries
        # corresponding to the queries of the overlapping frames
        
        overlapping_frames = query_init.get("frames", None)
        overlapping_clicks = query_init.get("clicks", None)

        if overlapping_frames is not None:
            num_overlapping_fg_queries = query_init["queries"][0].shape[0]
            for fr_idx in range(T):
                # for each overlapping query, add a learnable one per frame
                descriptors[fr_idx] = [repeat(self.no_click_query, "1 C -> 1 1 C") for _ in range(num_overlapping_fg_queries)]
                # record corresponding clicks
                clicks = overlapping_clicks[0].clone()
                clicks[:,3] = fr_idx/T
                normalized_clicks[fr_idx] = [n.squeeze(0) for n in torch.split(clicks, 1, dim=0)]

        # append new clicks
        for fg_coords in batched_fg_coords_list:
            y,x,obj_id,fr_idx,t = fg_coords

            for fr, desc, clks in zip(range(T), descriptors, normalized_clicks):
                if fr==fr_idx:
                    # normalize the click
                    clks.append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))

                    # extract query descriptor
                    clicks = torch.tensor([fg_coords], dtype=torch.float, device=device)
                    clicks = clicks[:,:2]
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
                    avg_click_query = torch.mean(torch.stack(click_queries, -1), dim = -1)
                    desc.extend(torch.split(avg_click_query, 1, dim=1))
                else:
                    desc.append(repeat(self.no_click_query, "1 C -> 1 1 C"))
                    clks.append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))
        
        if overlapping_frames is not None:
            num_overlapping_bg_queries = query_init["queries"][1].shape[0]
            if num_overlapping_bg_queries > 0:
                for fr_idx in range(T):
                    # for each overlapping bg query, add a learnable one per frame
                    descriptors[fr_idx].extend([repeat(self.no_click_query, "1 C -> 1 1 C") for _ in range(num_overlapping_bg_queries)])
                    # record corresponding clicks
                    clicks = overlapping_clicks[1].clone()
                    clicks[:,3] = fr_idx/T
                    normalized_clicks[fr_idx].extend([n.squeeze(0) for n in torch.split(clicks, 1, dim=0)])
        
        # background queries
        for bg_coords in batched_bg_coords_list:
            y,x,obj_id,fr_idx,t = bg_coords
            assert obj_id == -1
        
            for fr, desc, clks in zip(range(T), descriptors, normalized_clicks):
                if fr==fr_idx:
                    clks.append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))
        
                    clicks = torch.tensor([bg_coords], dtype=torch.float, device=device)
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
                    desc.extend(torch.split(avg_bg_query, 1, dim=1))
                else:
                    desc.append(repeat(self.no_click_query, "1 C -> 1 1 C"))
                    clks.append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))

        # at this point, in each frame, there is at least one query for 
        # each target present in that frame. Each click query in one frame 
        # has corresponding learnable queries in the other frames
        descriptors = [torch.cat(desc, dim=1) for desc in descriptors]
        descriptors = torch.cat(descriptors, dim=0)
        
        normalized_clicks = [torch.stack(clks).unsqueeze(0) for clks in normalized_clicks]
        normalized_clicks = torch.cat(normalized_clicks, dim=0).to(device)

        # update the number of queries per object
        num_queries_per_target = [0 for _ in range(N)]
        obj_ids_in_queries = normalized_clicks[0][:,2].to(torch.int)
        for obj_id in obj_ids_in_queries:
            if obj_id > 0:
                num_queries_per_target[obj_id - 1] += 1
            else:
                num_queries_per_target[-1] += 1
        
        return descriptors, normalized_clicks, num_queries_per_target

    
    def get_features_descriptors(self, fmap, point_coords_per_image):

        # fmap: 1xCxHxW 
        # point_coords_per_image: 1XQx2   

        y = point_sample(fmap, point_coords_per_image, align_corners=False) # 1xCxPoints
        
        return torch.permute(y, (0, 2, 1))