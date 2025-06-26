import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
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

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # learnable query for each object
        self.register_parameter("no_click_query", nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=True))
        
        nn.init.xavier_uniform_(self.no_click_query)
        
    
    def forward(
            self,
            features: Tensor,
            batched_fg_coords_list: List, 
            batched_bg_coords_list: List,
            num_clicks_per_object: List,
            norms: Tuple,
    ) -> Tensor:
        """
        For the given list of sampled clicks, extract queries from image features

        Queries belonging to the same frame are grouped together. In each frame, each object
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

        norm_h, norm_w, norm_t = norms
        
        # number of feature scales
        feature_levels = len(features)
        device = features[0][0].device
        T,_,h,w = features[-1].shape
        N = len(num_clicks_per_object[0]) + 1 # add bg
        H = float(h*8)
        W = float(w*8)
        
        descriptors = [[] for _ in range(T)]
        normalized_clicks = [[] for _ in range(T)]
        num_queries_per_object = [0 for _ in range(N)]

        # obtain queries one instance at a time, across all frames
        for fg_coords in batched_fg_coords_list:
            y,x,obj_id,fr_idx,t = fg_coords

            for fr, desc, clks in zip(range(T), descriptors, normalized_clicks):
                if fr==fr_idx:
                    # normalize the click
                    clks.append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))
                    num_queries_per_object[obj_id-1] += 1

                    # extract query descriptor
                    clicks = torch.tensor([fg_coords], dtype=torch.float, device=device)
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

                        nbd_features = self.get_features_descriptors(fmap_scale_fr, clicks.unsqueeze(0))    # 1,1,D
                        inst_queries.append(nbd_features)

                    # take the average of the features from multiple scales as the click query
                    avg_inst_query = torch.mean(torch.stack(inst_queries, -1), dim = -1)
                    desc.extend(torch.split(avg_inst_query, 1, dim=1))
                else:
                    desc.append(repeat(self.no_click_query, "1 C -> 1 1 C"))
                    clks.append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))
        
        # for inst_id in range(N):
        #     for fr_idx, fr_fg_coords in enumerate(batched_fg_coords_list):
        #         inst_fg_coords = fr_fg_coords[inst_id]
                
        #         if len(inst_fg_coords) > 0:
        #             # for each click on the instance in the current frame
        #             # add a learnable query in the other frames of the clip
                        
        #             for fr, desc, clks in zip(range(T), descriptors, normalized_clicks):
        #                 if fr==fr_idx:
        #                     for coords in inst_fg_coords:
        #                         # normalize the click
        #                         clks.append(torch.tensor([coords[0]/norm_h, coords[1]/norm_w, coords[2], coords[3]/T, coords[-1]/norm_t]))
        #                         num_queries_per_object[inst_id] += 1

        #                     # extract query descriptor
        #                     clicks = torch.tensor(inst_fg_coords, dtype=torch.float, device=device)
        #                     clicks = clicks[:,:2]
        #                     clicks[:,0]/=H
        #                     clicks[:,1]/=W
        #                     # invert (y,x) -> (x,y)
        #                     clicks = clicks.flip(-1)
                            
        #                     inst_queries = []
        #                     # extract click features in each scale of multi-res features
        #                     for i in range(feature_levels):
        #                         # feature maps at i-th feature scale
        #                         fmap_scale = features[i]
        #                         # map of particular frame at i-th feature scale
        #                         fmap_scale_fr = fmap_scale[fr_idx].unsqueeze(0)

        #                         nbd_features = self.get_features_descriptors(fmap_scale_fr, clicks.unsqueeze(0))    # 1,1,D
        #                         inst_queries.append(nbd_features)

        #                     # take the average of the features from multiple scales as the click query
        #                     avg_inst_query = torch.mean(torch.stack(inst_queries, -1), dim = -1)
        #                     desc.extend(torch.split(avg_inst_query, 1, dim=1))
        #                 else:
        #                     for coords in inst_fg_coords:
        #                         desc.append(repeat(self.no_click_query, "1 C -> 1 1 C"))
        #                         clks.append(torch.tensor([coords[0]/norm_h, coords[1]/norm_w, coords[2], fr/T, coords[-1]/norm_t]))
        #                         # num_queries_per_object[inst_id] += 1
        
        # background queries
        if len(batched_bg_coords_list) > 0: 
            for bg_coords in batched_bg_coords_list:
                y,x,obj_id,fr_idx,t = bg_coords
                assert obj_id == -1
            
                for fr, desc, clks in zip(range(T), descriptors, normalized_clicks):
                    if fr==fr_idx:
                        clks.append(torch.tensor([y/norm_h, x/norm_w, obj_id, fr/T, t/norm_t]))
                        num_queries_per_object[fr_idx][-1] += 1
            
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
                        num_queries_per_object[fr][-1] += 1

        # at this point, in each frame, there is at least one query for 
        # each object present in that frame. Each click query in one frame 
        # has corresponding learnable queries in the other frames
        descriptors = [torch.cat(desc, dim=1) for desc in descriptors]
        descriptors = torch.cat(descriptors, dim=0)
        
        normalized_clicks = [torch.stack(clks).unsqueeze(0) for clks in normalized_clicks]
        normalized_clicks = torch.cat(normalized_clicks, dim=0).to(device)
        
        return descriptors, normalized_clicks, num_queries_per_object


    def get_features_descriptors(self, fmap, point_coords_per_image):

        # fmap: 1xCxHxW 
        # point_coords_per_image: 1XQx2   

        y = point_sample(fmap, point_coords_per_image, align_corners=False) # 1xCxPoints
        
        return torch.permute(y, (0, 2, 1))