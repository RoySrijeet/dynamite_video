# # Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
"""
Various positional encodings for the transformer.
"""
import math

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        """
        Args:
            num_pos_feats: Number of positional features. Typically, it is set to half of the total 
                    embedding dimension since we later concatenate sine and cosine terms. Default: 64
            temperature: hyperparameter for temperature scaling. Default: 10000
            normalize: whether to scale the embeddings to the range [0,scale], ensuring they stay 
                    bounded. Default: False
            scale: scaling factor, if normalization is used. Default: None

        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale


    def forward_contiguous(self, x, mask=None):
        """
        Compute contiguous positional embedding across all images in the batch, 
        i.e., all frames of a clip

        Args:
            x: feature map of shape [T, D, h, w]
            mask: determines which spatial locations are valid for positional embedding computation
        
        Return:
            pos: positional embedding [Th, w, D]
        """
        if mask is None:
            mask = torch.zeros((x.size(0) * x.size(2), x.size(3)), device=x.device, dtype=torch.bool)   # Thxw
        not_mask = ~mask

        # relative spatial coordinates for each pixel
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        # scaling factor with exponential decay based on temperature
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3       # Th, D/2, w
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3       # Th, D/2, w
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x), dim=2)  # Th, D, w
        return pos
    
    
    def forward(self, x, mask=None, contiguous=False):
        """
        Compute positional embeddings for a 2D feature map

        Args:
            x: feature map of shape [T, D, h, w]
            contiguous: whether to generate conntiguous p.e. across frames of the batch
            mask: determines which spatial locations are valid for positional embedding computation

        Return:
            pos: positional embedding [T, D, h, w]
        """
        if contiguous:
            return self.forward_contiguous(x, mask)

        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask

        # relative spatial coordinates for each pixel
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
