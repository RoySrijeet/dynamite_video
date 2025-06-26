import fvcore.nn.weight_init as weight_init
import numpy as np
import random
import torch
import os

from collections import defaultdict
from einops import repeat, rearrange
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.memory import retry_if_cuda_oom

from dynamite_video.model.interactive_transformer.position_encoding import PositionEmbeddingSine, get_spatiotemporal_embeddings
from dynamite_video.model.interactive_transformer.descriptor_initializer import AvgClicksPoolingInitializer
from dynamite_video.model.interactive_transformer.utils import INTERACTIVE_TRANSFORMER_REGISTRY, MLP
from dynamite_video.model.interactive_transformer.encoder import Encoder
from dynamite_video.model.interactive_transformer.decoder import Decoder
from dynamite_video.training.train_utils import get_next_clicks


@INTERACTIVE_TRANSFORMER_REGISTRY.register()
class DynamiteInteractiveTransformer(nn.Module):

    _version = 2

    @configurable
    def __init__(
        self,
        in_channels,
        *,
        max_num_rounds: int,
        use_qqca,
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
        max_objects_to_refine: int,
        iou_threshold: float,
        refine_strategy:str
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
            max_objects_to_refine: num of objects to refine in each corrective round
        """
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        self.positional_embeddings = positional_embeddings
         # iterative
        self.max_num_rounds = max_num_rounds
        self.max_objects_to_refine = max_objects_to_refine
        self.iou_threshold = iou_threshold
        self.refine_strategy = refine_strategy

        self.num_static_bg_queries = num_static_bg_queries
        
        # Reverse Cross Attn
        self.use_decoder = use_decoder
        self.dec_layers = dec_layers
        self.dec_scale_factor = dec_scale_factor

        self.num_heads = nheads
        self.enc_layers = enc_layers
        self.use_qqca = use_qqca
        self.encoder = Encoder(hidden_dim, dim_feedforward, nheads, self.enc_layers, pre_norm, self.use_qqca)
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
        self.register_parameter("bg_query_pe", nn.Parameter(torch.zeros(hidden_dim), False))

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
        ret["use_qqca"] = cfg.MODEL.MASK_FORMER.QQCA

        # DECODER
        ret["use_decoder"] =  cfg.MODEL.MASK_FORMER.DECODER.USE_DECODER
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DECODER.DEC_LAYERS
        ret["dec_scale_factor"] = cfg.MODEL.MASK_FORMER.DECODER.DEC_SCALE_FACTOR

        # Iterative Pipeline
        ret["max_num_rounds"] = cfg.ITERATIVE.TRAIN.MAX_NUM_REFINEMENT_ROUNDS
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

        ret["max_objects_to_refine"] = cfg.CLICKER.TRAINING.MAX_NUM_INSTANCES_REFINED_PER_ROUND
        ret["iou_threshold"] = cfg.CLICKER.TRAINING.IOU_THRESHOLD
        ret["refine_strategy"] = cfg.CLICKER.TRAINING.REFINEMENT_STRATEGY
        return ret


    def forward(
            self, 
            data, 
            images, 
            objects_per_frame, 
            multi_scale_features, 
            mask_features, 
            num_clicks_per_object=None,
            fg_coords=None, 
            bg_coords=None, 
            max_timestamp=None,
            visualize=None,
            train_iter=None,
    ):
        """
        Forward pass of one video clip through the interactive transformer
        
        Args:
            data: input from dataloader, with all metadata
            images: [T, 3, H, W] tensors of the images in the clip (d2 ImageList)
            objects_per_frame: objects present in each frame of the clip
            multi_scale_features: list of frame features (T,C,H,W) extracted at different scale
            mask_features: mask features of the frames in the clip (T,C,H,W)
            num_clicks_per_object: list of click counts on each object, in each frame of the clip
            fg_coords: list of fg clicks on the frames of the clip
            bg_coords: list bg clicks on the frames of the clip
            max_timestamp: list of timestamps of the last clip on each frame of the clip
        """

        assert len(multi_scale_features) == self.num_feature_levels

        # extract memory features for Transformer (cross-)attention
        memory = []
        memory_pe = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            
            memory_pe_i = self.pe_layer(multi_scale_features[i], None)
            memory_pe.append(memory_pe_i.flatten(2))
            memory_i = self.input_proj[i](multi_scale_features[i])
            memory.append(memory_i.flatten(2) + self.level_embed.weight[i][None, :, None])
            
            # flatten NxCxHxW to HWxNxC
            memory_pe[-1] = memory_pe[-1].permute(2, 0, 1)  # TxDxhw -> hwxTxD
            memory[-1] = memory[-1].permute(2, 0, 1)        # TxDxhw -> hwxTxD

        # if visualize:
        #     visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/memory"
        #     torch.save(memory,      os.path.join(visualize_dir, f"memory_iter_{train_iter}.pth"))
        #     torch.save(memory_pe,   os.path.join(visualize_dir, f"memory_pe_iter_{train_iter}.pth"))
        
        if self.training:

            # number of corrective iterations
            num_rounds = random.randint(0, self.max_num_rounds)
            for i in range(num_rounds):

                # generate current queries, transformer forward pass
                prev_output, num_queries_per_object = self.iterative_batch_forward(multi_scale_features, 
                                                                                   memory, memory_pe, 
                                                                                   size_list, 
                                                                                   mask_features, 
                                                                                   fg_coords, bg_coords, 
                                                                                   num_clicks_per_object,
                                                                                   max_timestamp,
                                                                                   visualize=visualize, train_iter=train_iter)
                
                # segmentation mask from prediction logits
                processed_results = self.process_results(data, images, prev_output, objects_per_frame, num_queries_per_object, visualize, train_iter, i)

                # sample corrective clicks
                num_clicks_per_object, fg_coords, bg_coords, max_timestamp = get_next_clicks(data, 
                                                                                             processed_results, 
                                                                                             num_clicks_per_object,
                                                                                             fg_coords, bg_coords, 
                                                                                             max_timestamp, 
                                                                                             max_objects_to_refine=self.max_objects_to_refine,
                                                                                             iou_threshold=self.iou_threshold,
                                                                                             refine_strategy=self.refine_strategy,
                                                                                             visualize=visualize, train_iter=train_iter, round_num=i)
            
            # generate current queries, transformer forward pass
            outputs, num_queries_per_object = self.iterative_batch_forward(multi_scale_features, 
                                                                           memory, memory_pe, 
                                                                           size_list, 
                                                                           mask_features, 
                                                                           fg_coords, bg_coords, 
                                                                           num_clicks_per_object, 
                                                                           max_timestamp,
                                                                           visualize=visualize, train_iter=train_iter)
        else:
            # evaluation
            outputs, num_queries_per_object = self.iterative_batch_forward(multi_scale_features, 
                                                                           memory, memory_pe, 
                                                                           size_list, 
                                                                           mask_features, 
                                                                           fg_coords, bg_coords, 
                                                                           num_clicks_per_object, 
                                                                           max_timestamp)
        
        return outputs, num_queries_per_object

    
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
            output: decoder output, QxTxD
            mask_features: features from video frames, TxDxHxW
            attn_mask_target_size: target size of attention mask for next 
                feature scale, (h,w) tuple
        """

        decoder_output = self.layer_norm(output).transpose(0,1)
        mask_embed = self.mask_embed(decoder_output)
      
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features) # TxQxHxW

        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)   # TxQxhxw

        # boolean attention mask - (T*num_heads)xQx(hw)
        attn_mask = (attn_mask.sigmoid() < 0.5).repeat(self.num_heads,1,1,1).flatten(2).detach()

        return outputs_mask, attn_mask
      
    
    def iterative_batch_forward(
            self, 
            multi_scale_features, 
            memory, 
            memory_pe, 
            size_list, 
            mask_features,
            fg_coords,
            bg_coords, 
            num_clicks_per_object, 
            max_timestamp,
            visualize=None,
            train_iter=None
    ):
        """
        Prepare query descriptors and forward pass through Transformer
        """
        
        _, T, _ = memory[0].shape           # hw, T, D
        _, C, H, W = mask_features.shape
        device = multi_scale_features[0][0].device
        height = 4*H
        width = 4*W
        
        # generate query descriptors for input clicks
        (descriptors,                       # TxQxD
         normalized_clicks,                 # TxQxD
         num_queries_per_object) = self.query_descriptors_initializer(features=multi_scale_features,
                                                                    batched_fg_coords_list=fg_coords, 
                                                                    batched_bg_coords_list=bg_coords,
                                                                    num_clicks_per_object=num_clicks_per_object, 
                                                                    norms=(height, width, max(max_timestamp)),
                                                                )
        
        # positional embedding for queries
        query_embed = repeat(self.query_embed, "C -> Q T C", Q=descriptors.shape[1], T=T)   # QxTxD
        if self.positional_embeddings:
            pos_coord_embed = get_spatiotemporal_embeddings(normalized_clicks[:,:,[0,1,-1]].permute(1,0,2),
                                                            self.positional_embeddings, 
                                                            descriptors.shape[2])           # QxTxD'
            pos_coord_embed = self.ca_qpos_sine_proj(pos_coord_embed.to(query_embed.dtype)) # QxTxD
            query_embed = query_embed + pos_coord_embed                                     # QxTxD

        if visualize:
            visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/queries"
            torch.save(descriptors, os.path.join(visualize_dir, f"descriptors_iter_{train_iter}.pth"))
            torch.save(query_embed, os.path.join(visualize_dir, f"query_embed_iter_{train_iter}.pth"))
            torch.save(num_queries_per_object, os.path.join(visualize_dir, f"num_queries_per_object_iter_{train_iter}.pth"))
            torch.save(normalized_clicks, os.path.join(visualize_dir, f"normalized_click_iter_{train_iter}.pth"))

        # static background queries
        if self.use_static_bg_queries:
            static_bg_queries = repeat(self.static_bg_query, "Bg C -> T Bg C", T=T)
            descriptors = torch.cat((descriptors, static_bg_queries), dim=1)   # TxQxD
            static_bg_pe = repeat(self.static_bg_pe, "Bg C -> Bg T C", T=T)
            query_embed = torch.cat((query_embed, static_bg_pe), dim=0)        # QxTxD
            num_queries_per_object[-1] += static_bg_queries.shape[1]

            if visualize:
                visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/queries"
                torch.save(descriptors, os.path.join(visualize_dir, f"descriptors_w_static_bg_iter_{train_iter}.pth"))
                torch.save(query_embed, os.path.join(visualize_dir, f"query_embed_w_static_bg_iter_{train_iter}.pth"))
                torch.save(num_queries_per_object, os.path.join(visualize_dir, f"num_queries_per_object_w_static_bg_iter_{train_iter}.pth"))
        
        # if there's no bg query, remove from record
        if num_queries_per_object[-1] == 0:
            num_queries_per_object = num_queries_per_object[:-1]

        # total num queries per object across T frames
        if self.use_qqca == "masked":
            # if queries are batched instance-wise, consider the frame positions of the queries in the temporal domain
            inst_batched_query_embed = repeat(self.query_embed, "C -> Q N C", 
                                              Q=max(num_queries_per_object) * T, 
                                              N=len(num_queries_per_object))  # Q'xNxD
            # convert QxTx5 clicks to Q'xNx5
            inst_batched_clicks = self.get_object_batched_clicks(normalized_clicks.permute(1,0,2), num_queries_per_object)
            pos_coord_embed = get_spatiotemporal_embeddings(inst_batched_clicks[:,:,[0,1,3]],
                                                            self.positional_embeddings,
                                                            descriptors.shape[2])                        # Q'xNxD
            pos_coord_embed = self.ca_qpos_sine_proj(pos_coord_embed.to(inst_batched_query_embed.dtype)) # Q'xNxD
            if self.use_static_bg_queries:
                pos_coord_embed = torch.cat((pos_coord_embed, static_bg_pe.transpose(0,1)), dim=1)
            inst_batched_query_embed = inst_batched_query_embed + pos_coord_embed                        # Q'xNxD

        
        # pre-encoder prediction
        output = self.queries_nonlinear_projection(descriptors).permute(1,0,2)
        outputs_mask, attn_mask = self.forward_prediction_heads(output, 
                                                                mask_features, 
                                                                attn_mask_target_size=size_list[0])
        if visualize:
            visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/forward_prediction_heads"
            torch.save(outputs_mask,    os.path.join(visualize_dir, f"outputs_mask_pre_enc_iter_{train_iter}.pth"))
            torch.save(attn_mask,       os.path.join(visualize_dir, f"attn_mask_pre_enc_iter_{train_iter}.pth"))
        
        # store predicted mask after each layer, used in auxiliary loss
        predictions_mask = []
        predictions_mask.append(outputs_mask)
        
        # encoder
        for i in range(self.enc_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            
            #### CROSS-ATTENTION
            
            if visualize:
                visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/image_query_cross_attention"
                torch.save(output,                  os.path.join(visualize_dir, f"input_query_enc_layer_{i}_iter_{train_iter}.pth"))
                torch.save(query_embed,             os.path.join(visualize_dir, f"input_query_pe_enc_layer_{i}_iter_{train_iter}.pth"))
                torch.save(memory[level_index],     os.path.join(visualize_dir, f"input_memory_enc_layer_{i}_iter_{train_iter}.pth"))
                torch.save(memory_pe[level_index],  os.path.join(visualize_dir, f"input_memory_pe_enc_layer_{i}_iter_{train_iter}.pth"))
                torch.save(attn_mask,               os.path.join(visualize_dir, f"input_attn_mask_enc_layer_{i}_iter_{train_iter}.pth"))

            
            # cross-attention between image features and queries in each frame
            output, weights = self.encoder.cross_attention_layers[i](tgt=output,                     # QxTxD
                                                            memory=memory[level_index],     # (hw)xTxD
                                                            memory_mask=attn_mask,          # (T*#attn_heads)xQx(hw)
                                                            memory_key_padding_mask=None,   # here we do not apply masking on padded region
                                                            pos=memory_pe[level_index],     # (hw)xTxD pos emb for memory
                                                            query_pos=query_embed           # QxTxD pos emb for query
                                                        )
            if visualize:
                visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/image_query_cross_attention"
                torch.save(output,                  os.path.join(visualize_dir, f"output_query_enc_layer_{i}_iter_{train_iter}.pth"))
                torch.save(weights,                 os.path.join(visualize_dir, f"attn_weights_enc_layer_{i}_iter_{train_iter}.pth"))
                
                outputs_mask_inspection, _ = self.forward_prediction_heads(output, mask_features, 
                                                                            attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/inspection_query_query_cross_attention"
                torch.save(outputs_mask_inspection, os.path.join(visualize_dir, f"output_inspection_after_iqca_enc_layer_{i}_iter_{train_iter}.pth"))
        

            
            #### VANILLA QQCA
            
            # query-query cross-attention
            if self.use_qqca == "vanilla":
                Q,T,D = output.shape
                output, weights = self.encoder.query_query_cross_attention_layers[i](output.view(Q*T, 1, D),
                                                                            tgt_mask=None,
                                                                            tgt_key_padding_mask=None,
                                                                            query_pos=query_embed.view(Q*T, 1, D))
                output = output.view(Q,T,D)
                if visualize:
                    visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/vanilla_qqca"
                    torch.save(output,               os.path.join(visualize_dir, f"output_query_enc_layer_{i}_iter_{train_iter}.pth"))
                    torch.save(weights,              os.path.join(visualize_dir, f"attn_weights_enc_layer_{i}_iter_{train_iter}.pth"))
                    outputs_mask_inspection, _ = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                    torch.save(outputs_mask_inspection, os.path.join(visualize_dir, f"mask_inspection_after_qqca_enc_layer_{i}_iter_{train_iter}.pth"))
            
            
            
            #### MASKED QQCA
            
            
            if self.use_qqca == "masked":
                # cross-attention between object-specific queries of different frames
                inst_batched_query, qqca_mask = self.get_object_batched_query(output, num_queries_per_object)
                if visualize:
                    visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/masked_qqca"
                    torch.save(inst_batched_query,               os.path.join(visualize_dir, f"inst_batched_query_enc_layer_{i}_iter_{train_iter}.pth"))
                    torch.save(inst_batched_query_embed,         os.path.join(visualize_dir, f"inst_batched_query_embed_enc_layer_{i}_iter_{train_iter}.pth"))
                    torch.save(qqca_mask,                        os.path.join(visualize_dir, f"qqca_mask_enc_layer_{i}_iter_{train_iter}.pth"))
                
                padded_output, weights = self.encoder.query_query_cross_attention_layers[i](inst_batched_query,
                                                                                        tgt_mask=None,
                                                                                        tgt_key_padding_mask=qqca_mask.to(output.device),
                                                                                        query_pos=inst_batched_query_embed,
                                                                                    )
                output = self.get_frame_batched_query(output, padded_output, num_queries_per_object)
                if visualize:
                    visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/masked_qqca"
                    torch.save(padded_output,        os.path.join(visualize_dir, f"inst_batched_query_output_enc_layer_{i}_iter_{train_iter}.pth"))
                    torch.save(weights,              os.path.join(visualize_dir, f"attn_weights_enc_layer_{i}_iter_{train_iter}.pth"))
                    torch.save(output,               os.path.join(visualize_dir, f"output_query_enc_layer_{i}_iter_{train_iter}.pth"))
                    outputs_mask_inspection, _ = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                    torch.save(outputs_mask_inspection, os.path.join(visualize_dir, f"mask_inspection_after_qqca_enc_layer_{i}_iter_{train_iter}.pth"))
            

            #### SELF-ATTENTION

            if visualize:
                visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/self_attention"
                torch.save(output,                  os.path.join(visualize_dir, f"input_query_enc_layer_{i}_iter_{train_iter}.pth"))
                torch.save(query_embed,             os.path.join(visualize_dir, f"input_query_pe_enc_layer_{i}_iter_{train_iter}.pth"))
            
            # self-attention between queries within frame
            output, weights = self.encoder.self_attention_layers[i](output, 
                                                           tgt_mask=None, 
                                                           tgt_key_padding_mask=None,
                                                           query_pos=query_embed)
            if visualize:
                visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/self_attention"
                torch.save(output,  os.path.join(visualize_dir, f"output_query_enc_layer_{i}_iter_{train_iter}.pth"))
                torch.save(weights, os.path.join(visualize_dir, f"attn_weights_enc_layer_{i}_iter_{train_iter}.pth"))
                
                outputs_mask_inspection, _ = self.forward_prediction_heads(output, mask_features, 
                                                                            attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                torch.save(outputs_mask_inspection, os.path.join(visualize_dir, f"mask_inspection_after_sa_enc_layer_{i}_iter_{train_iter}.pth"))
            
            
            #### FFN
            
            output = self.encoder.ffn_layers[i](output)

            outputs_mask, attn_mask = self.forward_prediction_heads(output, 
                                                                    mask_features, 
                                                                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            if visualize:
                visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/ffn"
                torch.save(output,                  os.path.join(visualize_dir, f"output_query_enc_layer_{i}_iter_{train_iter}.pth"))
                torch.save(outputs_mask,            os.path.join(visualize_dir, f"outputs_mask_enc_layer_{i}_iter_{train_iter}.pth"))
                torch.save(attn_mask,               os.path.join(visualize_dir, f"attn_mask_enc_layer_{i}_{train_iter}.pth"))

            predictions_mask.append(outputs_mask)

        
        ### DECODER
        
        if self.use_decoder:
            if self.dec_scale_factor > 1:
                scale_factor = self.dec_scale_factor
                mask_features = F.interpolate(mask_features, scale_factor=scale_factor, mode='bilinear', align_corners=False)
           
            mask_features, weights = self.decoder((mask_features, output, query_embed))
            mask_features = rearrange(mask_features,"(H W) T C -> T C H W", H=H, W=W, T=T).contiguous()
            outputs_mask, attn_mask = self.forward_prediction_heads(output, 
                                                                    mask_features, 
                                                                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            if visualize:
                visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/decoder"
                torch.save(output,          os.path.join(visualize_dir, f"query_decoder_iter_{train_iter}.pth"))
                torch.save(mask_features,   os.path.join(visualize_dir, f"mask_features_decoder_iter_{train_iter}.pth"))
                torch.save(weights,         os.path.join(visualize_dir, f"attn_weights_decoder_iter_{train_iter}.pth"))

                visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/iterative_batch_forward/forward_prediction_heads"
                torch.save(outputs_mask,    os.path.join(visualize_dir, f"outputs_mask_decoder_iter_{train_iter}.pth"))
                torch.save(attn_mask,       os.path.join(visualize_dir, f"attn_mask_decoder_{train_iter}.pth"))
            
            predictions_mask.append(outputs_mask)

        out = {
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_mask)
        }
        return out, num_queries_per_object


    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
    
    
    def get_object_batched_clicks(
            self, 
            clicks, 
            num_queries_per_object
    ):
        """
        Convert Q,5 clicks into Q',N,5

        Args:
            clicks: click information
        """
        Q,T,D = clicks.shape
        # max num queries per object (across all frames)
        max_num_queries = max(num_queries_per_object) * T

        # split frame-wise clicks into object-wise queries
        object_wise_splits = torch.split(clicks, num_queries_per_object, dim=0)   # list of T,q_i,5
        object_wise_splits = [part.reshape(-1, D) for part in object_wise_splits]       # list of T*q_i,5

        # apply padding for batching
        pad = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0]).to(clicks.device)
        inst_batched_clicks = []
        for split_clicks in object_wise_splits:
            pad_len = max_num_queries - split_clicks.shape[0]
            if pad_len > 0:
                padding = pad.expand(pad_len, -1)
                padded_r = torch.cat([split_clicks, padding], dim=0)
            else:
                padded_r = split_clicks
            inst_batched_clicks.append(padded_r)

        inst_batched_clicks = torch.stack(inst_batched_clicks).transpose(0,1)

        return inst_batched_clicks

    
    def get_object_batched_query(
            self, 
            output, 
            num_queries_per_object
    ):
        """
        Convert Q,T,D query into Q',N,D query where:
        T: num of frames
        N: num of target objects + 1 BG
        Q: num of queries per frame = sum(q_1, q_2, ..., q_N)
        Q': max(q_i)
        """
        Q,T,D = output.shape
        
        # max num queries per object (across all frames)
        max_num_queries = max(num_queries_per_object) * T
        
        # split frame-wise queries into object-wise queries
        object_wise_splits = torch.split(output, num_queries_per_object, dim=0)   # list of T,q_i,D
        object_wise_splits = [part.reshape(-1, D) for part in object_wise_splits]       # list of T*q_i,D
        
        # keep a record of how many queries there were for each object
        # this will be used to create an attention mask
        orig_lengths = [r.shape[0] for r in object_wise_splits]
        
        # apply padding for batching
        inst_batched_query = []
        for split_query in object_wise_splits:
            pad_len = max_num_queries - split_query.shape[0]
            if pad_len > 0:
                padding = self.bg_query.expand(pad_len, -1)
                padded_r = torch.cat([split_query, padding], dim=0)
            else:
                padded_r = split_query
            inst_batched_query.append(padded_r)

        inst_batched_query = torch.stack(inst_batched_query).transpose(0,1)

        # attention mask
        qqca_mask = torch.arange(max_num_queries).expand(len(orig_lengths), max_num_queries) >= torch.tensor(orig_lengths).unsqueeze(1)

        return inst_batched_query, qqca_mask

    
    def get_frame_batched_query(
            self, 
            output, 
            padded_output, 
            num_queries_per_object
    ):
        """
        Convert Q',N,D query into Q,T,D
        """
        Q,T,D = output.shape
        padded_output = padded_output.permute(1, 0, 2)  # N,Q',D
        
        unpadded_chunks = []
        for i, q_size in enumerate(num_queries_per_object):
            q_len = T * q_size

            # i-th object, upto q_len tensors (rest were padding)
            if q_size > 0:
                block = padded_output[i, :q_len]            # T*q_i,D
                block = block.reshape(q_size, T, -1)        # q_i,T,D
                unpadded_chunks.append(block)

        # Concatenate all q_i blocks along Q
        restored = torch.cat(unpadded_chunks, dim=0)  # (Q, T, D)
        return restored

    
    def process_results(
            self, 
            data,
            images, 
            outputs, 
            objects_per_frame,
            num_queries_per_object,
            visualize=None,
            train_iter=None,
            round_num=None
    ):
        """
        Args:
            data: dataloader input
            images: [T, 3, H, W] tensors of the images in the clip (d2 ImageList)
            outputs: prediction 
            objects_per_frame: List of object IDs in the i-th frame
            num_queries_per_object: count of queries on each object in each frame
        """
        
        mask_pred_results = outputs["pred_masks"]   # [T,Q,H,W]
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del outputs

        # padding mask
        padding_mask = torch.logical_not(data["padding_mask"]).to(mask_pred_results.device)
        ignore_masks = torch.logical_not(torch.asarray(data["ignore_masks"])).to(torch.uint8).to(mask_pred_results.device)

        # objects in the whole clip
        seq_objects = sorted(list(set(x for ids in objects_per_frame for x in ids)))

        processed_results = []
        for mask_pred_per_image, objects_per_image, fr_ignore_mask in zip(mask_pred_results, objects_per_frame, ignore_masks):
            
            processed_r = retry_if_cuda_oom(self.interactive_object_inference)(mask_pred_per_image * padding_mask * fr_ignore_mask,
                                                                               objects_per_image, 
                                                                               num_queries_per_object,
                                                                               seq_objects)
            
            processed_results.append(processed_r * padding_mask * fr_ignore_mask)

        if visualize:
            visualize_dir = "/home/roy/REPOS/dynamite_video/debug/visualization/training/interactive_transformer/process_results"
            torch.save(mask_pred_results,       os.path.join(visualize_dir, f"upsampled_predictions_round_{round_num}_iter_{train_iter}.pth"))
            torch.save(padding_mask,            os.path.join(visualize_dir, f"padding_mask_round_{round_num}_iter_{train_iter}.pth"))
            torch.save(ignore_masks,            os.path.join(visualize_dir, f"ignore_masks_round_{round_num}_iter_{train_iter}.pth"))
            torch.save(objects_per_frame,       os.path.join(visualize_dir, f"objects_per_frame_round_{round_num}_iter_{train_iter}.pth"))
            torch.save(num_queries_per_object,  os.path.join(visualize_dir, f"num_queries_per_object_round_{round_num}_iter_{train_iter}.pth"))
            torch.save(processed_results,       os.path.join(visualize_dir, f"processed_results_round_{round_num}_iter_{train_iter}.pth"))
        
        return processed_results

    
    def interactive_object_inference(
            self, 
            mask_pred, 
            objects_per_image, 
            queries_per_object,
            seq_objects,
    ):
        """
        Given the raw predictions from Transformer, obtain binary segmentation masks

        Args:
            mask_pred: raw prediction from Transformer, TxQxHxW
            objects_per_image: list of object IDs in current frame
            queries_per_objects: count of queries on each object in current frame
            seq_objects: all objects present in the clip
        """

        H,W = mask_pred.shape[1:]
        temp_out = []
        splited_masks = torch.split(mask_pred, queries_per_object, dim=0)
        for m in splited_masks:
            if len(m)>0:
                temp_out.append(torch.max(m, dim=0).values)
        
        mask_pred = torch.stack(temp_out)       # (N+1)xHxW
        
        prob = torch.cat([torch.prod(1-mask_pred, dim=0, keepdim=True), mask_pred], 0).clamp(1e-7, 1-1e-7)
        logits = torch.log((prob /(1-prob)))
        logits = F.softmax(logits, dim=0)[1:]
        binary = (logits > 0.5).to(torch.uint8)
        
        binary_masks = torch.zeros((len(queries_per_object),H,W), dtype=torch.uint8)
        c = 0
        for i, q in enumerate(queries_per_object):
            if q>0:
                binary_masks[i][torch.where(binary[c]==1)] = 1
                c += 1
            
        return binary_masks.to(mask_pred.device)
        
        # mask_pred = torch.argmax(mask_pred,0)
        
        # m = []
        # for inst_id in seq_objects:
        #     if inst_id in objects_per_image:
        #         m.append((mask_pred == inst_id-1).float())
        #     else:
        #         m.append(torch.zeros(H,W).to(mask_pred.device))
        
        # mask_pred = torch.stack(m)
     
        # return mask_pred