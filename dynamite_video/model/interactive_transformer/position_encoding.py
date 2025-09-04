import math
import torch
from torch import nn


def get_query_positional_encodings(pos_tensor, positional_embeddings, hidden_dim=256):
    """
    Spatio-temporal positional embedding

    Args:
        pos_tensor: tensor of shape [Q,T,5] containing normalized clicks in the format [y,x,obj_id,fr_idx,click_timestamp]
        positional_embeddings: encoding dimensions, choose from ["2D", "3D", "4D", "5D"]
            2D: only spatial
            3D: + frame index
            4D: + object ID
            5D: + click timestamp
        hidden_dim: Transformer hidden dimension (int, default=256)
    """
    # num of positional features; typically set to half of the total embedding dimension since we later concatenate the different dimensions
    num_pos_feats = hidden_dim // 2
    # hyperparameter for temperature scaling
    temperature = 10000
    # scaling factor for normalization
    scale = 2 * math.pi
    # scaling frequencies
    dim_t = torch.arange(num_pos_feats, dtype=torch.float, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    
    # spatial positional encodings
    
    # spatial locations of clicks; already normalized, so only scale
    y_embed = pos_tensor[:, :, 0] * scale
    x_embed = pos_tensor[:, :, 1] * scale
    
    # scale by frequencies
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    
    # truncate negative values
    pos_x[:, :, 0::2][torch.where(pos_x[:, :, 0::2] < 0)] = 0.0
    pos_x[:, :, 1::2][torch.where(pos_x[:, :, 1::2] < 0)] = math.pi/2
    pos_y[:, :, 0::2][torch.where(pos_y[:, :, 0::2] < 0)] = 0.0
    pos_y[:, :, 1::2][torch.where(pos_y[:, :, 1::2] < 0)] = math.pi/2
    
    # apply alternating sin/cos
    pos_x = torch.stack(
        (
            pos_x[:, :, 0::2].sin(),
            pos_x[:, :, 1::2].cos()
        ), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (
            pos_y[:, :, 0::2].sin(),
            pos_y[:, :, 1::2].cos()
        ), dim=3
    ).flatten(2)

    if positional_embeddings == "2D":
        return torch.cat((pos_y, pos_x), dim=2) # Q,T,D
    
    # add frame index to positional encoding
    
    # frame index info; already normalized, so only scale
    f_embed = pos_tensor[:, :, 3] * scale
    # scale by frequencies
    pos_f = f_embed[:, :, None] / dim_t
    # truncate negative values
    pos_f[:, :, 0::2][torch.where(pos_f[:, :, 0::2] < 0)] = 0.0
    pos_f[:, :, 1::2][torch.where(pos_f[:, :, 1::2] < 0)] = math.pi/2
    # apply alternating sin/cos
    pos_f = torch.stack(
        (
            pos_f[:, :, 0::2].sin(),
            pos_f[:, :, 1::2].cos()
        ), dim=3
    ).flatten(2)
    
    if positional_embeddings == "3D":
        return torch.cat((pos_y, pos_x, pos_f), dim=2)  # Q,T,(3/2)*D
    
    # add object IDs to positional encoding

    # frame index info; already normalized, so only scale
    o_embed = pos_tensor[:, :, 2] * scale
    # scale by frequencies
    pos_o = o_embed[:, :, None] / dim_t
    # truncate negative values
    pos_o[:, :, 0::2][torch.where(pos_o[:, :, 0::2] < 0)] = 0.0
    pos_o[:, :, 1::2][torch.where(pos_o[:, :, 1::2] < 0)] = math.pi/2
    # apply alternating sin/cos
    pos_o = torch.stack(
        (
            pos_o[:, :, 0::2].sin(),
            pos_o[:, :, 1::2].cos()
        ), dim=3
    ).flatten(2)
    
    if positional_embeddings == "4D":
        return torch.cat((pos_y, pos_x, pos_f, pos_o), dim=2)  # Q,T,2*D
    
    # add click timestamps to positional encoding

    # click timestamps; already normalized, so only scale
    t_embed = pos_tensor[:, :, 4] * scale
    # scale by frequencies
    pos_t = t_embed[:, :, None] / dim_t
    # truncate negative values
    pos_t[:, :, 0::2][torch.where(pos_t[:, :, 0::2] < 0)] = 0.0
    pos_t[:, :, 1::2][torch.where(pos_t[:, :, 1::2] < 0)] = math.pi/2
    # apply alternating sin/cos
    pos_t = torch.stack(
        (
            pos_t[:, :, 0::2].sin(),
            pos_t[:, :, 1::2].cos()
        ), dim=3
    ).flatten(2)

    return torch.cat((pos_y, pos_x, pos_f, pos_o, pos_t), dim=2)  # Q,T,(5/2)*D



class PositionalEncoding(nn.Module):

    def __init__(
            self, 
            num_pos_feats=64,
            encoding_dims="spatial"
    ):
        """
        Args:
            num_pos_feats: Number of positional features. Typically, it is set to half of the total 
                    embedding dimension since we later concatenate sine and cosine terms. Default: 64
            encoding_dims: which dimensions are taken into account for encoding. Choose from -
                "spatial": only spatial encodings
                "spatio_temporal": spatial encodings + temporal position of the frame in the clip

        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.encoding_dims = encoding_dims
        
        # hyperparameter for temperature scaling
        self.temperature = 10000
        # scaling factor for normalization
        self.eps = 1e-6
        self.scale = 2 * math.pi

    
    def forward(self, x, mask=None):
        """
        Compute positional embeddings for a 2D feature map

        Args:
            x: feature map of shape [T, D, h, w]
            mask: determines which spatial locations are valid for positional embedding computation

        Return:
            pos: positional embedding [T, D, h, w]
        """

        T,_,H,W = x.shape
        if mask is None:
            mask = torch.zeros((T,H,W), device=x.device, dtype=torch.bool)
        not_mask = ~mask

        # relative spatial coordinates for each pixel
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        # normalize to scale the embeddings to the range [0,scale], ensuring they stay bounded
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        # denominator (scaling frequencies)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        # scale by frequencies
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # apply sin() to even positions, cos() to odd positions
        pos_x = torch.stack(
            (
                pos_x[:, :, :, 0::2].sin(),
                pos_x[:, :, :, 1::2].cos()
            ), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (
                pos_y[:, :, :, 0::2].sin(), 
                pos_y[:, :, :, 1::2].cos()
            ), dim=4
        ).flatten(3)
        
        if self.encoding_dims=="spatial":
            # concatenate only the spatial encodings
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        elif self.encoding_dims=="spatio_temporal":
            
            # relative temporal coordinate of each frame in the clip
            t_embed = torch.arange(T, device=x.device, dtype=torch.float32)  # (T,)
            t_embed = t_embed[:, None, None].expand(T, H, W)                 # (T, H, W)
            
            # normalize to scale the embeddings to the range [0,scale]
            t_embed = t_embed / (t_embed[-1] + self.eps) * self.scale
            
            # scale by frequencies
            pos_t = t_embed[:, :, :, None] / dim_t      # (T, H, W, d)
            
            # apply sin() to even positions, cos() to odd positions
            pos_t = torch.stack(
                (
                    pos_t[:, :, :, 0::2].sin(), 
                    pos_t[:, :, :, 1::2].cos()
                ), dim=4
            ).flatten(3)

            # concatenate spatial and temporal positional encodings
            pos = torch.cat((pos_y, pos_x, pos_t), dim=3).permute(0, 3, 1, 2)
        
        return pos

    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            f"num_pos_feats: {self.num_pos_feats}",
            f"encoding_dims: {self.encoding_dims}",
            f"temperature: {self.temperature}",
            f"normalization scale: {self.scale}",
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
