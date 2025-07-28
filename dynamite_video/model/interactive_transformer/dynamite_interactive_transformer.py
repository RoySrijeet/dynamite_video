import fvcore.nn.weight_init as weight_init
import torch

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
        max_targets_to_refine: int,
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
            max_targets_to_refine: num of targets to refine in each corrective round
        """
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        self.positional_embeddings = positional_embeddings
         # iterative
        self.max_num_rounds = max_num_rounds
        self.max_targets_to_refine = max_targets_to_refine
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

        ret["max_targets_to_refine"] = cfg.CLICKER.TRAINING.MAX_NUM_INSTANCES_REFINED_PER_ROUND
        ret["iou_threshold"] = cfg.CLICKER.TRAINING.IOU_THRESHOLD
        ret["refine_strategy"] = cfg.CLICKER.TRAINING.REFINEMENT_STRATEGY

        return ret


    def forward(
            self, 
            data, 
            images, 
            multi_scale_features, 
            mask_features, 
            num_clicks_per_target,
            fg_coords, 
            bg_coords, 
            max_timestamp
    ):
        """
        Forward pass of one video clip through the interactive transformer
        
        Args:
            data: input from dataloader, with all metadata
            images: [T, 3, H, W] tensors of the images in the clip (d2 ImageList)
            multi_scale_features: list of frame features (T,C,H,W) extracted at different scale
            mask_features: mask features of the frames in the clip (T,C,H,W)
            num_clicks_per_target: list of click counts on each target, in each frame of the clip
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
        
        
        if self.training:
            # iterative refinement in training
            
            # number of corrective iterations
            num_rounds = self.max_num_rounds    # TODO: #random.randint(0, self.max_num_rounds)
            for i in range(num_rounds):

                # generate current queries, transformer forward pass
                prev_output, num_queries_per_target = self.iterative_batch_forward(data, multi_scale_features, mask_features, 
                                                                                   memory, memory_pe, size_list, 
                                                                                   num_clicks_per_target,
                                                                                   fg_coords, bg_coords, max_timestamp)
                
                # segmentation mask from prediction logits
                processed_results = self.process_results(data, images, prev_output, len(num_clicks_per_target[0]), num_queries_per_target)

                # sample corrective clicks
                num_clicks_per_target, fg_coords, bg_coords, max_timestamp = get_next_clicks(data, processed_results, 
                                                                                             num_clicks_per_target,
                                                                                             fg_coords, bg_coords, max_timestamp, 
                                                                                             max_objects_to_refine=self.max_targets_to_refine,
                                                                                             iou_threshold=self.iou_threshold,
                                                                                             refine_strategy=self.refine_strategy)
            
        # generate current queries, transformer forward pass
        outputs, num_queries_per_target = self.iterative_batch_forward(data, multi_scale_features, mask_features, 
                                                                        memory, memory_pe, size_list, 
                                                                        num_clicks_per_target,
                                                                        fg_coords, bg_coords, max_timestamp)
        return outputs, num_queries_per_target

    
    def forward_prediction_heads(
            self, 
            output, 
            mask_features, 
            attn_mask_target_size,
            orig_clicks=None
    ):
        """
        Obtain predicted mask from queries and mask features.
        Use predicted mask to generate attention mask for next feature scale.
        
        Args:
            output: decoder output, QxTxD
            mask_features: features from video frames, TxDxHxW
            attn_mask_target_size: target size of attention mask for next 
                feature scale, (h,w) tuple
        """

        decoder_output = self.layer_norm(output).transpose(0,1)
        mask_embed = self.mask_embed(decoder_output)
        
        # mask prediction
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features) # TxQxHxW

        # attention mask
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)   # TxQxhxw
        # boolean attention mask
        attn_mask = (attn_mask.sigmoid() < 0.5)
        
        if orig_clicks is not None:
            # do not mask for the learnable queries
            T,Q,_,_ = outputs_mask.shape
            for fr_idx in range(T):
                for i in range(Q):
                    if i < len(orig_clicks) and fr_idx != orig_clicks[i][3]:
                        attn_mask[fr_idx][i] = False
        
        # (T*num_heads)xQx(hw)
        attn_mask = attn_mask.repeat(self.num_heads,1,1,1).flatten(2).detach()
        return outputs_mask, attn_mask
      
    
    def iterative_batch_forward(
            self, 
            data,
            multi_scale_features, 
            mask_features,
            memory, 
            memory_pe, 
            size_list, 
            num_clicks_per_target,
            fg_coords,
            bg_coords, 
            max_timestamp
    ):
        """
        Prepare query descriptors and forward pass through Transformer
        """
        
        T, _, H, W = mask_features.shape
        height,width = data["images"].shape[-2:]
        
        # QUERY INITIALIZATION
        (descriptors,                       # TxQxD
         normalized_clicks,                 # TxQxD
         num_queries_per_target) = self.query_descriptors_initializer(features=multi_scale_features,
                                                                    batched_fg_coords_list=fg_coords, 
                                                                    batched_bg_coords_list=bg_coords,
                                                                    num_clicks_per_target=num_clicks_per_target, 
                                                                    norms=(height, width, max(max_timestamp))
                                                                )
        
        # SPATIO-TEMPORAL EMBEDDING
        query_embed = repeat(self.query_embed, "C -> Q T C", Q=descriptors.shape[1], T=T)   # QxTxD
        if self.positional_embeddings:
            pos_coord_embed = get_spatiotemporal_embeddings(normalized_clicks[:,:,[0,1,-1]].permute(1,0,2),
                                                            self.positional_embeddings, 
                                                            descriptors.shape[2])           # QxTxD'
            pos_coord_embed = self.ca_qpos_sine_proj(pos_coord_embed.to(query_embed.dtype)) # QxTxD
            query_embed = query_embed + pos_coord_embed                                     # QxTxD

        # STATIC BG QUERIES
        if self.use_static_bg_queries:
            static_bg_queries = repeat(self.static_bg_query, "Bg C -> T Bg C", T=T)
            descriptors = torch.cat((descriptors, static_bg_queries), dim=1)                # TxQxD
            static_bg_pe = repeat(self.static_bg_pe, "Bg C -> Bg T C", T=T)
            query_embed = torch.cat((query_embed, static_bg_pe), dim=0)                     # QxTxD
            # add bg queries to the count
            num_queries_per_target[-1] += self.num_static_bg_queries
            # add proxy bg clicks to the click
            normalized_clicks = torch.cat([normalized_clicks, torch.full((T, self.num_static_bg_queries, 5), -1.0, device=normalized_clicks.device, dtype=normalized_clicks.dtype)], dim=1)

        # TODO: jerry-built; if there's no bg query, remove from record
        if num_queries_per_target[-1] == 0:
            num_queries_per_target = num_queries_per_target[:-1]
        
        
        # SPATIO_TEMPORAL EMBEDDING FOR MASKED QQCA
        if self.use_qqca == "masked":
            # if queries are batched target-wise, consider the frame positions of the queries in the temporal domain
            tgt_batched_query_embed = repeat(self.query_embed, "C -> Q N C", 
                                              Q=max(num_queries_per_target) * T, 
                                              N=len(num_queries_per_target))  # Q'xNxD
            # convert QxTx5 clicks to Q'xNx5
            tgt_batched_clicks = self.get_target_batched_clicks(normalized_clicks.permute(1,0,2), num_queries_per_target)
            pos_coord_embed = get_spatiotemporal_embeddings(tgt_batched_clicks[:,:,[0,1,3]],
                                                            self.positional_embeddings,
                                                            descriptors.shape[2])                        # Q'xNxD
            pos_coord_embed = self.ca_qpos_sine_proj(pos_coord_embed.to(tgt_batched_query_embed.dtype)) # Q'xNxD
            tgt_batched_query_embed = tgt_batched_query_embed + pos_coord_embed                        # Q'xNxD

        
        # MLP
        output = self.queries_nonlinear_projection(descriptors).permute(1,0,2)  # QxTxD
        
        # PRE-ENCODER PREDICTION
        outputs_mask, attn_mask = self.forward_prediction_heads(output, 
                                                                mask_features, 
                                                                attn_mask_target_size=size_list[0],
                                                                orig_clicks=fg_coords+bg_coords)
        # store predicted mask after each layer, later used in auxiliary loss
        predictions_mask = []
        predictions_mask.append(outputs_mask)
        
        # ENCODER
        for i in range(self.enc_layers):
            # encoder layers alternate between multi-scale features
            level_index = i % self.num_feature_levels
            
            # un-mask completely masked attention masks
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # IMAGE-QUERY CROSS ATTENTION between queries and image features (intra-frame)
            output = self.encoder.cross_attention_layers[i](tgt=output,             # QxTxD
                                                    memory=memory[level_index],     # (hw)xTxD
                                                    memory_mask=attn_mask,          # (T*#attn_heads)xQx(hw)
                                                    memory_key_padding_mask=None,
                                                    pos=memory_pe[level_index],
                                                    query_pos=query_embed)
            
            # QUERY-QUERY CROSS ATTENTION between queries (inter-frame)
            tgt_batched_query, qqca_mask = self.get_target_batched_query(output, num_queries_per_target)
            padded_output = self.encoder.query_query_cross_attention_layers[i](tgt_batched_query,
                                                                                tgt_mask=None,
                                                                                tgt_key_padding_mask=qqca_mask,
                                                                                query_pos=tgt_batched_query_embed)
            output = self.get_frame_batched_query(output, padded_output, num_queries_per_target)
            
            # SELF-ATTENTION between queries of the same frame (intra-frame)
            output = self.encoder.self_attention_layers[i](output, 
                                                                tgt_mask=None, 
                                                                tgt_key_padding_mask=None,
                                                                query_pos=query_embed)
            
            # FFN
            output = self.encoder.ffn_layers[i](output)
            outputs_mask, attn_mask = self.forward_prediction_heads(output, 
                                                                    mask_features, 
                                                                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_mask.append(outputs_mask)

        # DECODER
        if self.use_decoder:
            if self.dec_scale_factor > 1:
                scale_factor = self.dec_scale_factor
                mask_features = F.interpolate(mask_features, scale_factor=scale_factor, mode='bilinear', align_corners=False)
           
            mask_features = self.decoder((mask_features, output, query_embed))
            mask_features = rearrange(mask_features,"(H W) T C -> T C H W", H=H, W=W, T=T).contiguous()
            outputs_mask, attn_mask = self.forward_prediction_heads(output, 
                                                                    mask_features, 
                                                                    attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_mask.append(outputs_mask)

        out = {
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_mask)
        }
        
        return out, num_queries_per_target


    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
    
    
    def get_target_batched_clicks(
            self, 
            clicks, 
            num_queries_per_target
    ):
        """
        Convert Q,5 clicks into Q',N,5

        Args:
            clicks: click information
        """
        Q,T,D = clicks.shape
        # max num queries per target (across all frames)
        max_num_queries = max(num_queries_per_target) * T

        # split frame-wise clicks into target-wise queries
        target_wise_splits = torch.split(clicks, num_queries_per_target, dim=0)   # list of T,q_i,5
        target_wise_splits = [part.reshape(-1, D) for part in target_wise_splits]       # list of T*q_i,5

        # apply padding for batching
        pad = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0]).to(clicks.device)
        tgt_batched_clicks = []
        for split_clicks in target_wise_splits:
            pad_len = max_num_queries - split_clicks.shape[0]
            if pad_len > 0:
                padding = pad.expand(pad_len, -1)
                padded_r = torch.cat([split_clicks, padding], dim=0)
            else:
                padded_r = split_clicks
            tgt_batched_clicks.append(padded_r)

        tgt_batched_clicks = torch.stack(tgt_batched_clicks).transpose(0,1)

        return tgt_batched_clicks

    
    def get_target_batched_query(
            self, 
            output, 
            num_queries_per_target
    ):
        """
        Convert Q,T,D query into Q',N,D query where:
        T: num of frames
        N: num of target targets + 1 BG
        Q: num of queries per frame = sum(q_1, q_2, ..., q_N)
        Q': max(q_i)
        """
        Q,T,D = output.shape
        
        # max num queries per target (across all frames)
        max_num_queries = max(num_queries_per_target) * T
        
        # split frame-wise queries into target-wise queries
        target_wise_splits = torch.split(output, num_queries_per_target, dim=0)   # list of T,q_i,D
        target_wise_splits = [part.reshape(-1, D) for part in target_wise_splits]       # list of T*q_i,D
        
        # keep a record of how many queries there were for each target
        # this will be used to create an attention mask
        orig_lengths = [r.shape[0] for r in target_wise_splits]
        
        # apply padding for batching
        tgt_batched_query = []
        for split_query in target_wise_splits:
            pad_len = max_num_queries - split_query.shape[0]
            if pad_len > 0:
                padding = self.bg_query.expand(pad_len, -1)
                padded_r = torch.cat([split_query, padding], dim=0)
            else:
                padded_r = split_query
            tgt_batched_query.append(padded_r)

        tgt_batched_query = torch.stack(tgt_batched_query).transpose(0,1)

        # attention mask
        qqca_mask = torch.arange(max_num_queries).expand(len(orig_lengths), max_num_queries) >= torch.tensor(orig_lengths).unsqueeze(1)

        return tgt_batched_query, qqca_mask.to(output.device)

    
    def get_frame_batched_query(
            self, 
            output, 
            padded_output, 
            num_queries_per_target
    ):
        """
        Convert Q',N,D query into Q,T,D
        """
        Q,T,D = output.shape
        padded_output = padded_output.permute(1, 0, 2)  # N,Q',D
        
        unpadded_chunks = []
        for i, q_size in enumerate(num_queries_per_target):
            q_len = T * q_size

            # i-th target, upto q_len tensors (rest were padding)
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
            num_targets,
            num_queries_per_target
    ):
        """
        Args:
            data: dataloader input
            images: [T, 3, H, W] tensors of the images in the clip (d2 ImageList)
            outputs: prediction 
            num_targets: num of targets present in the clip
            num_queries_per_target: count of queries on each target in each frame
        """
        
        mask_pred_results = outputs["pred_masks"]   # [T,Q,H,W]
        # upsample to original resolution
        mask_pred_results = F.interpolate(mask_pred_results, size=(images.tensor.shape[-2], images.tensor.shape[-1]), mode="bilinear", align_corners=False,)
        del outputs

        # padding mask
        padding_mask = torch.logical_not(data["padding_mask"]).to(mask_pred_results.device)
        processed_results = []
        for mask_pred_per_image in mask_pred_results:
            processed_r = retry_if_cuda_oom(self.interactive_mask_inference)(mask_pred_per_image * padding_mask,
                                                                               num_targets, num_queries_per_target)
            processed_results.append(processed_r * padding_mask)
    
        return processed_results

    
    def interactive_mask_inference(
            self, 
            mask_pred, 
            num_targets,
            num_queries_per_target
    ):
        """
        Given the raw predictions from Transformer, obtain binary segmentation masks

        Args:
            mask_pred: raw prediction from Transformer, QxHxW
            num_targets: num of targets present in the clip
            num_queries_per_target: count of queries on each target in current frame
        """

        temp_out = []
        splited_masks = torch.split(mask_pred, num_queries_per_target, dim=0)
        for m in splited_masks:
            if len(m)>0:
                temp_out.append(torch.max(m, dim=0).values)
        
        mask_pred = torch.stack(temp_out)       # (N+1)xHxW
        mask_pred = torch.argmax(mask_pred,0)

        m = []
        for tgt_id in range(num_targets):
            m.append((mask_pred == tgt_id).float())
        mask_pred = torch.stack(m)
     
        return mask_pred