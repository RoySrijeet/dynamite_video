import fvcore.nn.weight_init as weight_init
import numpy as np
import random
import torch

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
        num_objects_to_refine: int,
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
            num_objects_to_refine: num of objects to refine in each corrective round
        """
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        self.positional_embeddings = positional_embeddings
         # iterative
        self.max_num_rounds = max_num_rounds

        self.num_static_bg_queries = num_static_bg_queries
        self.num_objects_to_refine = num_objects_to_refine
        
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

        ret["num_objects_to_refine"] = cfg.CLICKER.TRAINING.MAX_NUM_INSTANCES_REFINED_PER_ROUND
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
            max_timestamp=None
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
            
            memory_pe.append(self.pe_layer(multi_scale_features[i], None).flatten(2))
            memory.append(self.input_proj[i](multi_scale_features[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            
            # flatten NxCxHxW to HWxNxC
            memory_pe[-1] = memory_pe[-1].permute(2, 0, 1)  # TxDxhw -> hwxTxD
            memory[-1] = memory[-1].permute(2, 0, 1)        # TxDxhw -> hwxTxD


        if self.training:

            # number of corrective iterations
            num_rounds = random.randint(0, self.max_num_rounds)
            for i in range(num_rounds):

                # generate current queries, transformer forward pass
                prev_output, num_queries_per_object = self.iterative_batch_forward(multi_scale_features, 
                                                                                   memory, 
                                                                                   memory_pe, 
                                                                                   size_list, 
                                                                                   mask_features, 
                                                                                   fg_coords, 
                                                                                   bg_coords, 
                                                                                   max_timestamp)
                
                # segmentation mask from prediction logits
                processed_results = self.process_results(images, prev_output, data["padding_mask"], objects_per_frame, num_queries_per_object)

                # sample corrective clicks
                num_clicks_per_object, fg_coords, bg_coords, max_timestamp = get_next_clicks(data, 
                                                                                             processed_results, 
                                                                                             num_clicks_per_object,
                                                                                             fg_coords, 
                                                                                             bg_coords, 
                                                                                             max_timestamp, 
                                                                                             num_objects_to_refine=self.num_objects_to_refine)
            
            # generate current queries, transformer forward pass
            outputs, num_queries_per_object = self.iterative_batch_forward(multi_scale_features, 
                                                                           memory, 
                                                                           memory_pe, 
                                                                           size_list, 
                                                                           mask_features, 
                                                                           fg_coords, 
                                                                           bg_coords, 
                                                                           max_timestamp)
        else:
            # evaluation
            outputs, num_queries_per_object = self.iterative_batch_forward(multi_scale_features, 
                                                                           memory, 
                                                                           memory_pe, 
                                                                           size_list, 
                                                                           mask_features, 
                                                                           fg_coords, 
                                                                           bg_coords, 
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
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

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
            max_timestamp
    ):
        """
        Prepare query descriptors and forward pass through Transformer
        """
        
        _, T, _ = memory[0].shape           # hw, T, D
        B, C, H, W = mask_features.shape
        device = multi_scale_features[0][0].device
        height = 4*H
        width = 4*W
        
        # generate query descriptors for input clicks
        descriptors, normalized_click_coords, num_queries_per_object = self.query_descriptors_initializer(
                                                                            features=multi_scale_features,
                                                                            batched_fg_coords_list=fg_coords, 
                                                                            batched_bg_coords_list=bg_coords, 
                                                                            norms=(height, width, max(max_timestamp)),
                                                                        ) # TxQxD, TxQx3, TxN
        
        # pad descriptors of each frame so that they all have same length
        max_queries = max([desc.shape[1] for desc in descriptors])
        for i, desc in enumerate(descriptors):
            if self.use_static_bg_queries:
                bg_queries = repeat(self.bg_query, "C -> 1 L C", L=max_queries-desc.shape[1])
            else:
                bg_queries = repeat(self.bg_query, "C -> 1 L C", L=max_queries+1-desc.shape[1])
            descriptors[i] = torch.cat((descriptors[i], bg_queries), dim=1)

            clks = normalized_click_coords[i]
            if len(clks) < max_queries:
                diff = max_queries-len(clks)
                normalized_click_coords[i].extend([torch.tensor([-1.0, -1.0, -1.0])] * diff)
                num_queries_per_object[i][-1] += diff
        
        descriptors = torch.cat(descriptors, dim=0)  # TxQxD
        normalized_click_coords = [torch.stack(clks).unsqueeze(0) for clks in normalized_click_coords]
        normalized_click_coords = torch.cat(normalized_click_coords, dim=0).to(device)  # TxQx3

        # positional embedding for queries
        query_embed = repeat(self.query_embed, "C -> Q N C", N=T, Q=descriptors.shape[1])  # QxTxD
        if self.positional_embeddings:
            # spatio-temporal embedding
            pos_coord_embed = get_spatiotemporal_embeddings(normalized_click_coords.permute(1,0,2), self.positional_embeddings, descriptors.shape[2]) # QxTxD'
            pos_coord_embed = self.ca_qpos_sine_proj(pos_coord_embed.to(query_embed.dtype)) # QxTxD
            query_embed = query_embed + pos_coord_embed # QxTxD

        # static bg query
        if self.use_static_bg_queries:
            static_bg_pe = repeat(self.static_bg_pe, "Bg C -> Bg N C", N=T)
            query_embed = torch.cat((query_embed, static_bg_pe), dim=0)      # QxTxD
            static_bg_queries = repeat(self.static_bg_query, "Bg C -> N Bg C", N=T)
            descriptors = torch.cat((descriptors, static_bg_queries), dim=1)   # TxQxD
            for i in range(len(num_queries_per_object)):
                num_queries_per_object[i][-1] += static_bg_queries.shape[1]

        # for each object, store where corresponding queries are located
        object_to_indices = self.get_object_indexing_in_query(num_queries_per_object)
        
        # prepare query tensor
        output = self.queries_nonlinear_projection(descriptors).permute(1,0,2)

        predictions_mask = []
        # pre-transformer prediction
        outputs_mask, attn_mask = self.forward_prediction_heads(output, 
                                                                mask_features, 
                                                                attn_mask_target_size=size_list[0])
        predictions_mask.append(outputs_mask)

        # encoder
        for i in range(self.enc_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # cross-attention between image features and queries in each frame
            output = self.encoder.cross_attention_layers[i](tgt=output,                     # QxTxD
                                                            memory=memory[level_index],     # (hw)xTxD
                                                            memory_mask=attn_mask,          # (T*#attn_heads)xQx(hw)
                                                            memory_key_padding_mask=None,   # here we do not apply masking on padded region
                                                            pos=memory_pe[level_index],     # (hw)xTxD pos emb for memory
                                                            query_pos=query_embed           # QxTxD pos emb for query
                                                        )

            if self.use_qqca == "vanilla_before_msa":
                Q,T,D = output.shape
                output = self.encoder.query_query_cross_attention_layers[i](output.view(Q*T, D),
                                                                            tgt_mask=None,
                                                                            tgt_key_padding_mask=None,
                                                                            query_pos=query_embed.view(Q*T, D),
                                                                        )
                output = output.view(Q,T,D)
            if self.use_qqca == "masked_before_msa":
                # cross-attention between object-specific queries of different frames
                inst_batched_query, inst_batched_query_embed, inst_batched_pad_mask = self.get_object_batched_query(output, query_embed, object_to_indices, num_queries_per_object)
                padded_output = self.encoder.query_query_cross_attention_layers[i](inst_batched_query,
                                                                                tgt_mask=None,
                                                                                tgt_key_padding_mask=inst_batched_pad_mask,
                                                                                query_pos=inst_batched_query_embed,
                                                                            )
                output = self.get_frame_batched_query(output, padded_output, object_to_indices)
            
            # self-attention between queries within frame
            output = self.encoder.self_attention_layers[i](output, 
                                                           tgt_mask=None, 
                                                           tgt_key_padding_mask=None,
                                                           query_pos=query_embed
                                                        )
            
            if self.use_qqca == "vanilla_after_msa":
                Q,T,D = output.shape
                output = self.encoder.query_query_cross_attention_layers[i](output.view(Q*T, D),
                                                                            tgt_mask=None,
                                                                            tgt_key_padding_mask=None,
                                                                            query_pos=query_embed.view(Q*T, D),
                                                                        )
                output = output.view(Q,T,D)
            if self.use_qqca == "masked_after_msa":
                # cross-attention between object-specific queries of different frames
                inst_batched_query, inst_batched_query_embed, inst_batched_pad_mask = self.get_object_batched_query(output, query_embed, object_to_indices, num_queries_per_object)
                padded_output = self.encoder.query_query_cross_attention_layers[i](inst_batched_query,
                                                                                tgt_mask=None,
                                                                                tgt_key_padding_mask=inst_batched_pad_mask,
                                                                                query_pos=inst_batched_query_embed,
                                                                            )
                output = self.get_frame_batched_query(output, padded_output, object_to_indices)
            
            # FFN
            output = self.encoder.ffn_layers[i](output)

            outputs_mask, attn_mask = self.forward_prediction_heads(output, 
                                                                    mask_features, 
                                                                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])

            predictions_mask.append(outputs_mask)

        # decoder
        if self.use_decoder:
            if self.dec_scale_factor > 1:
                scale_factor = self.dec_scale_factor
                mask_features = F.interpolate(mask_features, scale_factor=scale_factor, mode='bilinear', align_corners=False)
           
            mask_features = self.decoder((mask_features, output, query_embed))
            mask_features = rearrange(mask_features,"(H W) B C -> B C H W", H=H, W=W, B=B).contiguous()
            outputs_mask, attn_mask = self.forward_prediction_heads(output, 
                                                                    mask_features, 
                                                                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
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
    
    
    def get_object_indexing_in_query(self, num_queries_per_object):
        """
        Given the num of queries per object in each frame, return a mapping of 
        each object to the query and frame index in the Q,T,D query tensor

        Args:
            num_queries_per_object: num of queries per object, np.ndarray [T, N]
        """
        object_to_indices = defaultdict(list)
        for fr_idx, num_queries_at_frame in enumerate(num_queries_per_object):
            q_offset = 0
            for inst_id, q_count in enumerate(num_queries_at_frame):
                for local_idx in range(q_count):
                    global_q_idx = q_offset + local_idx
                    object_to_indices[inst_id].append((global_q_idx, fr_idx))
                q_offset += q_count
        return object_to_indices
    

    def get_object_batched_query(
            self, 
            output, 
            query_embed, 
            object_to_indices, 
            num_queries_per_object
    ):
        """
        Convert Q,T,D query into Q',N,D

        Args:
            output: current query state, torch.Tensor [Q,T,D]
            query_embed: query positional embedding, torch.Tensor [Q,T,D]
            object_to_indices: mapping of each object and its position in `output`, dict
            num_queries_per_object: num of queries per object, np.ndarray [T, N]
        """
        
        device = output.device
        D = output.shape[-1]
        N = len(object_to_indices)
        
        # max num of queries for an object
        max_num_queries = max(np.sum(np.asarray(num_queries_per_object), axis=0))
        
        # store Q',D for each of the objects + BG
        object_batched_query = torch.full((N, max_num_queries, D), fill_value=0.0, device=device)   # Q'xNxD
        object_batched_query_embed = torch.full_like(object_batched_query, fill_value=0.0)          # Q'xNxD
        inst_batched_pad_mask = torch.zeros((N, max_num_queries), dtype=torch.bool, device=device)  # NxQ'
        
        for i, (_, indices) in enumerate(object_to_indices.items()):
            q_indices, t_indices = zip(*indices)
            q_indices = torch.tensor(q_indices)
            t_indices = torch.tensor(t_indices)

            inst_query = output[q_indices, t_indices]
            inst_query_embed = query_embed[q_indices, t_indices]
            q_len = inst_query.shape[0]
            
            object_batched_query[i, :q_len] = inst_query
            object_batched_query_embed[i, :q_len] = inst_query_embed
            inst_batched_pad_mask[i, q_len:] = True
        
        return object_batched_query.transpose(0,1), object_batched_query_embed.transpose(0,1), inst_batched_pad_mask
    

    def get_frame_batched_query(
            self, 
            output, 
            padded_output, 
            object_to_indices
    ):
        """
        Convert Q',N,D query into Q,T,D

        Args:
            output: query state before query-query attention, torch.Tensor [Q,T,D]
            padded_output: current query state, after query-query attention, torch.Tensor [Q',N,D]
            object_to_indices: mapping of each object and its position in `output`, dict
        """
        split_output = torch.split(padded_output, 1, dim=1)
        for inst_id, indices in object_to_indices.items():
            q_indices, t_indices = zip(*indices)
            q_indices = torch.tensor(q_indices)
            t_indices = torch.tensor(t_indices)

            output[q_indices, t_indices] = split_output[inst_id][:len(q_indices)].squeeze(1)
        return output

    
    def process_results(
            self, 
            images, 
            outputs, 
            padding_mask,
            objects_per_frame,
            num_queries_per_object
    ):
        """
        Args:
            images: [T, 3, H, W] tensors of the images in the clip (d2 ImageList)
            outputs: prediction 
            padding_mask: padding, [H,W]
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
        padding_mask = torch.logical_not(padding_mask).to(mask_pred_results.device)

        # objects in the whole clip
        seq_objects = sorted(list(set(x for ids in objects_per_frame for x in ids)))

        processed_results = []
        for mask_pred_per_image, objects_per_image, queries_per_object in zip(mask_pred_results, objects_per_frame, num_queries_per_object):
            
            processed_r = retry_if_cuda_oom(self.interactive_object_inference)(mask_pred_per_image * padding_mask, 
                                                                               objects_per_image, 
                                                                               queries_per_object, 
                                                                               seq_objects)
            
            processed_results.append(processed_r * padding_mask)

        return processed_results

    
    def interactive_object_inference(
            self, 
            mask_pred, 
            objects_per_image, 
            queries_per_object,
            seq_objects
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
            if len(m) == 0:
                temp_out.append(torch.zeros(H,W).to(mask_pred.device))
            else:
                temp_out.append(torch.max(m, dim=0).values)
        
        mask_pred = torch.stack(temp_out)       # (N+1)xHxW
        mask_pred = torch.argmax(mask_pred,0)
        
        m = []
        for inst_id in seq_objects:
            if inst_id in objects_per_image:
                m.append((mask_pred == inst_id-1).float())
            else:
                m.append(torch.zeros(H,W).to(mask_pred.device))
        
        mask_pred = torch.stack(m)
     
        return mask_pred