# Adapted by Amit Rana from: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py

import copy
import time as timer
import einops
import random
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch

from torch import nn, Tensor
from torch.nn import functional as F
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.config import configurable
from detectron2.layers import Conv2d
from einops import repeat
from .position_encoding import PositionEmbeddingSine
from .descriptor_initializer import AvgClicksPoolingInitializer
from dynamite_video.training.utils.train_utils import get_next_clicks, get_spatiotemporal_embeddings
from .utils import INTERACTIVE_TRANSFORMER_REGISTRY, MLP
from .encoder import Encoder
from .decoder import Decoder

@INTERACTIVE_TRANSFORMER_REGISTRY.register()
class DynamiteInteractiveTransformer(nn.Module):

    _version = 2

    @configurable
    def __init__(
        self,
        in_channels,
        *,
        max_num_interactions: int,
        use_decoder, 
        dec_layers,
        dec_scale_factor,
        use_static_bg_queries: bool,
        num_static_bg_queries: int,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        enc_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        positional_embeddings: str,
    ):
        """
        Args:
            in_channels: channels of the input features
            use_decoder: whether to use decoder
            dec_layers: number of decoder layers
            dec_scale_factor: scaling factor for mask_features before using in decoder
            use_static_bg_queries: whether to use learned background queries
            num_static_bg_queries: number of learned background queries
            hidden_dim: Transformer feature dimension
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            positional_embeddings: type of positonal embeddings for clicks coordinates 
        """
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        self.positional_embeddings = positional_embeddings
         # iterative
        self.max_num_interactions = max_num_interactions

        self.num_static_bg_queries = num_static_bg_queries
        
        # Reverse Cross Attn
        self.use_decoder = use_decoder
        self.dec_layers = dec_layers
        self.dec_scale_factor = dec_scale_factor

        self.num_heads = nheads
        self.enc_layers = enc_layers
        self.encoder = Encoder(hidden_dim, dim_feedforward, nheads, self.enc_layers, pre_norm)
        if self.use_decoder:
            self.decoder = Decoder(hidden_dim, nheads, self.dec_layers, pre_norm)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.query_descriptors_initializer = AvgClicksPoolingInitializer(hidden_dim)
        
        self.queries_nonlinear_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if self.positional_embeddings == "spatio_temporal":
            self.ca_qpos_sine_proj = nn.Linear(3*(hidden_dim//2), hidden_dim)
        elif self.positional_embeddings in ["temporal","spatial"]:
            self.ca_qpos_sine_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # learnable query p.e.
        self.register_parameter("query_embed", nn.Parameter(torch.zeros(hidden_dim), True))
       
        self.use_static_bg_queries = use_static_bg_queries
        if self.use_static_bg_queries:
            self.register_parameter("static_bg_pe", nn.Parameter(torch.zeros(self.num_static_bg_queries, hidden_dim), True))
            self.register_parameter("static_bg_query", nn.Parameter(torch.zeros(self.num_static_bg_queries,hidden_dim), True))
        self.register_parameter("bg_query", nn.Parameter(torch.zeros(hidden_dim), False))

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self._reset_parameters()

    
    def _reset_parameters(self):
        nn.init.normal_(self.query_embed)
        if self.use_static_bg_queries:
            nn.init.normal_(self.static_bg_pe)
            nn.init.xavier_uniform_(self.static_bg_query)

    
    @classmethod
    def from_config(cls, cfg, in_channels):
        ret = {}
        ret["in_channels"] = in_channels
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # DECODER
        ret["use_decoder"] =  cfg.MODEL.MASK_FORMER.DECODER.USE_DECODER
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DECODER.DEC_LAYERS
        ret["dec_scale_factor"] = cfg.MODEL.MASK_FORMER.DECODER.DEC_SCALE_FACTOR

        # Iterative Pipeline
        ret["max_num_interactions"] = cfg.ITERATIVE.TRAIN.MAX_NUM_INTERACTIONS
        ret["positional_embeddings"] = cfg.ITERATIVE.TRAIN.POSITIONAL_EMBED

        ret["use_static_bg_queries"] = cfg.ITERATIVE.TRAIN.USE_STATIC_BG_QUERIES
        ret["num_static_bg_queries"] = cfg.ITERATIVE.TRAIN.NUM_STATIC_BG_QUERIES
        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.ENC_LAYERS >= 1
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        return ret


    def forward(
            self, 
            data, 
            images, 
            num_instances, 
            multi_scale_features, 
            mask_features, 
            num_clicks_per_object=None,
            fg_coords=None, 
            bg_coords=None, 
            max_timestamp=None
    ):
        """
        Forward pass of one video clip through the interactive transformer
        
        Args:
            data: input from dataloader, with all metadata
            images: [T, 3, H, W] tensors of the images in the clip (d2 ImageList)
            num_instances: number of instances in each frame of the clip
            multi_scale_features: list of frame features (T,C,H,W) extracted at different scale
            mask_features: mask features of the frames in the clip (T,C,H,W)
            num_clicks_per_object: list of click counts on each instance, in each frame of the clip
            fg_coords: list of fg clicks on the frames of the clip
            bg_coords: list bg clicks on the frames of the clip
            max_timestamp: list of timestamps of the last clip on each frame of the clip
        """

        # x is a list of multi-scale feature
        assert len(multi_scale_features) == self.num_feature_levels
        
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])

            pos_i = self.pe_layer(multi_scale_features[i], None).flatten(2) # TxDxhw
            pos_i = pos_i.contiguous().view(-1, pos_i.shape[1]).clone() # THWxD
            pos.append(pos_i)
            
            src_i = self.input_proj[i](multi_scale_features[i]).flatten(2) + self.level_embed.weight[i][None, :, None]  # TxDxhw
            src_i = src_i.contiguous().view(-1, src_i.shape[1]).clone() # THWxD
            src.append(src_i)


        if self.training:
            prev_output = None
            num_iters = random.randint(0, self.max_num_interactions)   # TODO - reset
            # num_iters = random.randint(1, self.max_num_interactions)

            for i in range(num_iters):
                prev_output = self.iterative_batch_forward(multi_scale_features, src, pos, size_list, 
                                                            mask_features, fg_coords, bg_coords, max_timestamp
                )
                
                processed_results = self.process_results(data, images, prev_output, num_instances, num_clicks_per_object)
                                
                next_coords_info = get_next_clicks(data, processed_results, i+1, num_clicks_per_object, fg_coords, 
                                                    bg_coords, max_timestamp=max_timestamp)
                
                
                (num_clicks_per_object,  fg_coords, bg_coords, max_timestamp) = next_coords_info
                       
                        
            outputs = self.iterative_batch_forward(multi_scale_features, src, pos, size_list, mask_features, fg_coords, 
                                                    bg_coords, max_timestamp
                    )
        else:
            outputs = self.iterative_batch_forward(multi_scale_features, src, pos, size_list, mask_features, fg_coords, bg_coords,
                                                   max_timestamp)
        return outputs, num_clicks_per_object

    
    def forward_prediction_heads(
            self, 
            output, 
            mask_features, 
            attn_mask_target_size
    ):
        """
        Obtain predicted mask from decoder output and mask features.
        Use predicted mask to generate attention mask for next feature scale.
        
        Args:
            output: decoder output, QxD
            mask_features: features from video frames, TxDxHxW
            attn_mask_target_size: target size of attention mask for next 
                feature scale, (h,w) tuple
        """

        decoder_output = self.layer_norm(output)
        mask_embed = self.mask_embed(decoder_output)
      
        outputs_mask = torch.einsum("qc,bchw->bqhw", mask_embed, mask_features) # TxQxHxW

        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)   # TxQxhxw
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.

        # TxQxhxw -(sigmoid)-> TxQxhxw -(flatten)-> TxQx(hw) -unsqueeze-> Tx1xQx(hw) -(repeat)-> TxMxQx(hw) -(flatten)-> (TM)xQx(hw)    [M - num attn heads]
        # attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()    # (T*M)xQx(hw)
        
        # T,Q,h,w -(sigm)> T,Q,h,w -(flat)> T,Q,(hw) -(transpose)> T,(hw),Q -(flat)> (Thw),Q -(unsqueeze)> 1,(Thw),Q -(repeat)> M,(Thw),Q
        attn_mask = (attn_mask.sigmoid().flatten(2).transpose(1,2).flatten(0,1).unsqueeze(0).repeat(self.num_heads,1,1) < 0.5).bool()    # M,(Thw),Q
        attn_mask = attn_mask.transpose(1,2).detach()   # M,Q,Thw

        return outputs_mask, attn_mask

      
    
    def iterative_batch_forward(
            self, 
            x, 
            src, 
            pos, 
            size_list, 
            mask_features,
            fg_coords=None, 
            bg_coords=None, 
            max_timestamp=None
    ):
        """
        Meta
        """

        B, C, H, W = mask_features.shape
        height = 4*H
        width = 4*W
        
        # generate query descriptors for input clicks
        descriptors, normalized_click_coords = self.query_descriptors_initializer(
                                                    x, 
                                                    fg_coords, 
                                                    bg_coords, 
                                                    (height, width), 
                                                    max_timestamp=max_timestamp
                                                ) # QxD, Qx3
        query_embed = repeat(self.query_embed, "C -> Q C", Q=descriptors.shape[0]) # QxD

        if self.positional_embeddings:
            pos_coord_embed = get_spatiotemporal_embeddings(normalized_click_coords, self.positional_embeddings, descriptors.shape[1]) # QxD'
            pos_coord_embed = self.ca_qpos_sine_proj(pos_coord_embed.to(query_embed.dtype)) # QxD

            query_embed = query_embed + pos_coord_embed # QxD

        if self.use_static_bg_queries:
            query_embed = torch.cat((query_embed, self.static_bg_pe), dim=0)      # QxD
            descriptors = torch.cat((descriptors, self.static_bg_query), dim=0)   # QxD
    
        output = self.queries_nonlinear_projection(descriptors)
        predictions_mask = []
       
        # prediction heads on learnable query features
        outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_mask.append(outputs_mask)

        for i in range(self.enc_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.encoder.cross_attention_layers[i](
                                                            tgt=output,                     # QxD
                                                            memory=src[level_index],        # (hw)xTxD
                                                            memory_mask=attn_mask,          # (T*#attn_heads)xQx(hw)
                                                            memory_key_padding_mask=None,   # here we do not apply masking on padded region
                                                            pos=pos[level_index],           # (hw)xTxD pos emb for memory
                                                            query_pos=query_embed           # QxD pos emb for query
                                                        )

            output = self.encoder.self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.encoder.ffn_layers[i](
                output
            )

            outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_mask.append(outputs_mask)


        if self.use_decoder:
            if self.dec_scale_factor > 1:
                scale_factor = self.dec_scale_factor
                mask_features = F.interpolate(mask_features, scale_factor=scale_factor, mode='bilinear', align_corners=False)
           
            mask_features = self.decoder((mask_features, output, query_embed))
            mask_features = einops.rearrange(mask_features,"(B H W) C -> B C H W", H=H, W=W, B=B).contiguous()
            outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_mask.append(outputs_mask)

        out = {
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_mask)
        }
        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
    
    
    def process_results(
            self, 
            data, 
            images, 
            outputs, 
            num_instances, 
            num_clicks_per_object
    ):
        """
        Args:
            data: input from dataloader for current clip
            images: [T, 3, H, W] tensors of the images in the clip (d2 ImageList)
            outputs: prediction 
            num_instances: List [n_1, n_2, ..., n_T] where n_i is the #instances in the i-th frame
            num_clicks_per_object: count of clicks on each instance in each frame
        """
        
        mask_pred_results = outputs["pred_masks"]   # [T,C,H,W]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del outputs

        # padding mask
        padding_mask = torch.from_numpy(np.logical_not(data["padding_mask"])).to(mask_pred_results.device)
        processed_results = []

        for mask_pred_per_image, inst_per_image, clicks_per_image in zip(mask_pred_results, num_instances, num_clicks_per_object):
            mask_pred_per_image = mask_pred_per_image * padding_mask
            
            instance_r = retry_if_cuda_oom(self.interactive_instance_inference)(mask_pred_per_image, inst_per_image, clicks_per_image)
            processed_results.append(instance_r)

        return processed_results


    def interactive_instance_inference(self, mask_pred, num_instances, num_clicks_per_object):
        
        # assert len(num_clicks_per_object) == num_instances
        num_clicks_per_object_copy = copy.deepcopy(num_clicks_per_object)
        # handle zero clicks 
        for i in range(len(num_clicks_per_object_copy)):
            if num_clicks_per_object_copy[i] == 0:
                num_clicks_per_object_copy[i]+=1
        num_clicks_per_object_copy.append(mask_pred.shape[0]-sum(num_clicks_per_object_copy))
        
        temp_out = []
        if num_clicks_per_object_copy[-1] == 0:
            splited_masks = torch.split(mask_pred, num_clicks_per_object_copy[:-1], dim=0)
        else:
            splited_masks = torch.split(mask_pred, num_clicks_per_object_copy, dim=0)
        for m in splited_masks:
            temp_out.append(torch.max(m, dim=0).values)
        
        mask_pred = torch.stack(temp_out)
        mask_pred = torch.argmax(mask_pred,0)
        
        if num_instances > 0:
            if num_instances > 25:
                raise
            m = []
            for i in range(num_instances):
                m.append((mask_pred == i).float())
            
            mask_pred = torch.stack(m)
        else:
            assert mask_pred.ndim == 2
     
        return mask_pred
    


# def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
#         decoder_output = self.layer_norm(output)
#         decoder_output = decoder_output.transpose(0, 1)

#         mask_embed = self.mask_embed(decoder_output)
      
#         outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

#         # NOTE: prediction is of higher-resolution
#         # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
#         attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
#         # must use bool type
#         # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
#         attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
#         attn_mask = attn_mask.detach()

#         return outputs_mask, attn_mask

# def iterative_batch_forward(self, x, src, pos, size_list, mask_features,
#                                 fg_coords=None, bg_coords=None, max_timestamp=None):

#         _, bs, _ = src[0].shape
#         B, C, H, W = mask_features.shape
#         height = 4*H
#         width = 4*W

        
#         # original
#         # descriptors = self.query_descriptors_initializer(x, fg_coords, bg_coords)
#         # max_queries_batch = max([desc.shape[1] for desc in descriptors])
#         # for i, desc in enumerate(descriptors):
#         #     if self.use_static_bg_queries:
#         #         bg_queries = repeat(self.bg_query, "C -> 1 L C", L=max_queries_batch-desc.shape[1])
#         #     else:
#         #         bg_queries = repeat(self.bg_query, "C -> 1 L C", L=max_queries_batch+1-desc.shape[1])
#         #     descriptors[i] = torch.cat((descriptors[i], bg_queries), dim=1)
#         # output = torch.cat(descriptors, dim=0)  # TxQxD
#         #query_embed = repeat(self.query_embed, "C -> Q N C", N=bs, Q=output.shape[1]) # TxQxD

#         # if self.positional_embeddings:
#         #     normalized_click_coords = get_pos_tensor_coords(fg_coords, bg_coords,
#         #                                             output.shape[1], height, width, output.device, max_timestamp=max_timestamp
#         #                     ) # TxQx3
#         #     pos_coord_embed = get_spatiotemporal_embeddings(normalized_click_coords.permute(1,0,2), self.positional_embeddings) # Q x bs x C
#         #     pos_coord_embed = self.ca_qpos_sine_proj(pos_coord_embed.to(query_embed.dtype))
            
#         #     query_embed = query_embed + pos_coord_embed

#         # if self.use_static_bg_queries:
#         #     static_bg_pe = repeat(self.static_bg_pe, "Bg C -> Bg N C", N=bs)        # Bg = num static bg queries (default: 9)
#         #     query_embed = torch.cat((query_embed,static_bg_pe),dim=0)
#         #     static_bg_queries = repeat(self.static_bg_query, "Bg C -> N Bg C", N=bs)
#         #     output = torch.cat((output,static_bg_queries), dim=1)
    
#         # # NxQxC -> QxNxC
#         # output = self.queries_nonlinear_projection(output).permute(1,0,2)
#         # # query positional embedding QxNxC
        
#         # # query_embed = None
#         # predictions_mask = []
       
#         # prediction heads on learnable query features
#         outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
#         predictions_mask.append(outputs_mask)

#         for i in range(self.enc_layers):
#             level_index = i % self.num_feature_levels
#             attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
#             # attention: cross-attention first
#             output = self.encoder.cross_attention_layers[i](
#                 output, src[level_index],
#                 memory_mask=attn_mask,
#                 memory_key_padding_mask=None,  # here we do not apply masking on padded region
#                 pos=pos[level_index], query_pos=query_embed
#             )

#             output = self.encoder.self_attention_layers[i](
#                 output, tgt_mask=None,
#                 tgt_key_padding_mask=None,
#                 query_pos=query_embed
#             )
            
#             # FFN
#             output = self.encoder.ffn_layers[i](
#                 output
#             )

#             outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
#             predictions_mask.append(outputs_mask)


#         if self.use_decoder:
#             if self.dec_scale_factor > 1:
#                 scale_factor = self.dec_scale_factor
#                 mask_features = F.interpolate(mask_features, scale_factor=scale_factor, mode='bilinear', align_corners=False)
           
#             mask_features = self.decoder((mask_features, output, query_embed))
#             mask_features = einops.rearrange(mask_features,"(H W) B C -> B C H W", H=H, W=W, B=B).contiguous()
#             outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
#             predictions_mask.append(outputs_mask)

#         out = {
#             'pred_masks': predictions_mask[-1],
#             'aux_outputs': self._set_aux_loss(predictions_mask)
#         }
#         return out

