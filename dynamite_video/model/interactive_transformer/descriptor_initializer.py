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
            normalized_clicks_coords: space-time normalized clicks corresponding to each descriptor
        """

        norm_h, norm_w, norm_t = norms
        
        # number of feature scales
        feature_levels = len(features)
        device = features[0][0].device
        T,_,h,w = features[-1].shape
        N = len(batched_fg_coords_list[0])
        H = float(h*8)
        W = float(w*8)
        
        descriptors = []
        normalized_clicks = []
        num_queries_per_object = np.zeros((T, N+1)).astype('int')
        
        # stack queries for each frame
        for fr_idx, fr_fg_coords in enumerate(batched_fg_coords_list):

            # foreground queries of current frame
            fr_fg_queries = []
            fr_fg_normalized_clicks = []
            
            # stack queries for each instance in the frame
            for inst_id, inst_fg_coords in enumerate(fr_fg_coords):
                
                # always add a learnable query
                fr_fg_queries.append(repeat(self.no_click_query, "1 C -> 1 1 C"))
                fr_fg_normalized_clicks.append(torch.tensor([-1.0, -1.0, -1.0]))
                num_queries_per_object[fr_idx][inst_id] += 1

                if len(inst_fg_coords) > 0:
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

                        nbd_features = self.get_features_descriptors(fmap_scale_fr, clicks.unsqueeze(0))    # 1,1,D
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


        # NOTE: the implementation below is slower - bg clicks are empty more often than not
        # for fr_idx, fr_fg_coords, fr_bg_coords in zip(range(T), batched_fg_coords_list, batched_bg_coords_list):

        #     fr_descriptors = []
        #     normalized_clicks.append([])
        #     clicks = []
        #     insert_index = []
            
        #     # stack queries for each object in the frame
        #     for inst_id, inst_fg_coords in enumerate(fr_fg_coords):
                
        #         # always add a learnable query for each object
        #         fr_descriptors.append(repeat(self.no_click_query, "1 C -> 1 1 C"))
        #         normalized_clicks[-1].append(torch.tensor([-1.0, -1.0, -1.0]))
        #         num_queries_per_object[fr_idx][inst_id] += 1

        #         # collect fg clicks, if available
        #         if len(inst_fg_coords) == 0:
        #             continue
                
        #         # each click is normalized, and corresponding point features are computed once per frame (along with bg) later
        #         clicks.extend(inst_fg_coords)
        #         for coords in inst_fg_coords:
        #             insert_index.append(len(fr_descriptors))
        #             fr_descriptors.append([])
        #             normalized_clicks[-1].append(torch.tensor([coords[0]/norm_h, coords[1]/norm_w, coords[-1]/norm_t]))
        #             num_queries_per_object[fr_idx][inst_id] += 1
                
        #     # collect bg clicks
        #     if len(fr_bg_coords) > 0:
        #         clicks.extend(fr_bg_coords)
        #         for coords in fr_bg_coords:
        #             insert_index.append(len(fr_descriptors))
        #             fr_descriptors.append([])
        #             normalized_clicks[-1].append(torch.tensor([coords[0]/norm_h, coords[1]/norm_w, coords[-1]/norm_t]))
        #             num_queries_per_object[fr_idx][-1] += 1

        #     if len(clicks) > 0:
            
        #         # `clicks` contains both fg and bg clicks
        #         clicks = torch.tensor(clicks, dtype=torch.float, device=device)
        #         # extract and scale spatial coordinates
        #         clicks = clicks[:,:2]
        #         clicks[:,0]/=H
        #         clicks[:,1]/=W
        #         # invert (y,x) -> (x,y)
        #         clicks = clicks.flip(-1)
                
        #         fr_queries_per_scale = []
        #         # extract click features in each scale of multi-res features
        #         for i in range(feature_levels):
        #             # feature map of particular frame at i-th scale
        #             fmap_scale_fr = features[i][fr_idx].unsqueeze(0)
        #             # point features at current scale
        #             nbd_features = point_sample(fmap_scale_fr, clicks.unsqueeze(0), align_corners=False)
        #             nbd_features = torch.permute(nbd_features, (0, 2, 1))
        #             # split per query
        #             fr_queries_per_scale.append(torch.split(nbd_features, 1, dim=1))
                
        #         fr_queries_per_inst = list(zip(*fr_queries_per_scale))
        #         for idx, clk_features in zip(insert_index, fr_queries_per_inst):
        #             cat_features = torch.stack(clk_features, dim=-1)
        #             fr_descriptors[idx] = torch.mean(cat_features, dim=-1)

            
        #     descriptors.append(fr_descriptors)
        
        # descriptors = [torch.cat(desc, dim=1) for desc in descriptors]
        
        # return descriptors, normalized_clicks, num_queries_per_object.tolist()


    def get_features_descriptors(self, fmap, point_coords_per_image):

        # fmap: 1xCxHxW 
        # point_coords_per_image: 1XQx2   

        y = point_sample(fmap, point_coords_per_image, align_corners=False) # 1xCxPoints
        
        return torch.permute(y, (0, 2, 1))