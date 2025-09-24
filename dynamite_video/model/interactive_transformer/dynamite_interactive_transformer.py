import fvcore.nn.weight_init as weight_init
import random
import torch
import warnings

from einops import repeat, rearrange
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.memory import retry_if_cuda_oom

from dynamite_video.model.interactive_transformer.position_encoding import PositionalEncoding, get_query_positional_encodings
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
        in_channels: int,
        *,
        max_num_rounds: int,
        use_decoder: bool, 
        dec_layers: int,
        dec_scale_factor: float,
        use_static_bg_queries: bool,
        num_static_bg_queries: int,
        hidden_dim: int,
        nheads: int,
        dim_feedforward: int,
        enc_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_projection: bool,
        kv_positional_encoding: str,
        q_positional_encoding: str,
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
            enforce_input_projection: add input project 1x1 conv even if input
                channels and hidden dim is identical
            kv_positional_encoding: type of positonal embeddings for multi-scale image features
            q_positional_encoding: type of positonal embeddings for clicks coordinates
            max_targets_to_refine: num of targets to refine in each corrective round
            iou_threshold: refine a target until threshold is reached
            refine_strategy: which target objects are chosen during rounding for refinement, between ["random", "worst"]
        """
        super().__init__()

        ### POSITIONAL ENCODINGS ###
        
        # positional encodings for image features
        self.kv_positional_encoding = kv_positional_encoding
        self.pe_layer = PositionalEncoding(
            num_pos_feats=hidden_dim // 2,
            encoding_dims=self.kv_positional_encoding
        )
        # projection MLP head for positional encodings
        if self.kv_positional_encoding == "spatio_temporal":
            self.kv_pos_enc_proj = nn.Linear(3*(hidden_dim//2), hidden_dim)
        elif self.kv_positional_encoding == "spatial":
            self.kv_pos_enc_proj = nn.Linear(hidden_dim, hidden_dim)
        else: # TODO - decide on the default behaviour
            warnings.warn(f"Desired type of image-feature positional encoding {self.kv_positional_encoding} \
                          is not supported. Using default behaviour: 'spatio_temporal'")
            self.kv_positional_encoding = "spatio_temporal"
            self.kv_pos_enc_proj = nn.Linear(3*(hidden_dim//2), hidden_dim)

        # positional encodings for click queries [2D, 3D, 4D, 5D]
        self.q_positional_encoding = q_positional_encoding
        # projection MLP head for positional encodings
        num_pe_dims = int(self.q_positional_encoding[0])
        # each encoding dimension is given (hidden_dim // 2) num of positional features
        self.q_pos_enc_proj = nn.Linear(num_pe_dims * (hidden_dim // 2), hidden_dim)

        ### IMAGE FEATURE PROJECTION HEADS ###

        # num of feature levels in multi-scale image features extracted by pixel decoder
        self.num_feature_levels = 3
        
        # kv feature projection heads to match image feature channels with Transformer hidden dim
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_projection:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        
        # level embedding to impart scale identity - some knowledge about which scale the features come from
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        ### INTERACTIVE TRANSFORMER ###

        # encoder
        self.num_heads = nheads
        self.enc_layers = enc_layers
        self.encoder = Encoder(hidden_dim, dim_feedforward, nheads, self.enc_layers, pre_norm)
        # decoder
        self.use_decoder = use_decoder
        self.dec_layers = dec_layers
        self.dec_scale_factor = dec_scale_factor
        if self.use_decoder:
            self.decoder = Decoder(hidden_dim, nheads, self.dec_layers, pre_norm)
        # forward prediction head
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        ### QUERIES ###

        # query descriptor initializer from clicks
        self.query_descriptors_initializer = AvgClicksPoolingInitializer(hidden_dim)
        # projection head for raw query descriptors
        self.queries_nonlinear_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # learnable query positional encodings
        self.register_parameter("query_embed", nn.Parameter(torch.zeros(hidden_dim), True))
       
        ### STATIC BACKGROUND QUERIES ###

        self.use_static_bg_queries = use_static_bg_queries
        self.num_static_bg_queries = num_static_bg_queries
        if self.use_static_bg_queries:
            # NOTE: static bg queries are trainable parameters
            self.register_parameter("static_bg_pe", nn.Parameter(torch.zeros(self.num_static_bg_queries, hidden_dim), True))
            self.register_parameter("static_bg_query", nn.Parameter(torch.zeros(self.num_static_bg_queries,hidden_dim), True))
        
        # padding queries, used to pad QQCA queries and their p.e., **not trainable**
        self.register_parameter("pad_query", nn.Parameter(torch.zeros(hidden_dim), False))
        self.register_parameter("pad_query_pe", nn.Parameter(torch.zeros(hidden_dim), False))

        ### ROUDNING ###

        self.max_num_rounds = max_num_rounds
        self.max_targets_to_refine = max_targets_to_refine
        self.iou_threshold = iou_threshold
        self.refine_strategy = refine_strategy
        
        self._reset_parameters()
    

    def _reset_parameters(self):
        nn.init.normal_(self.query_embed)
        if self.use_static_bg_queries:
            nn.init.normal_(self.static_bg_pe)
            nn.init.xavier_uniform_(self.static_bg_query)

    
    @classmethod
    def from_config(cls, cfg, in_channels):
        ret = {}

        # positional encodings
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["kv_positional_encoding"] = cfg.MODEL.MASK_FORMER.KV_POSITIONAL_EMBED
        ret["q_positional_encoding"] = cfg.MODEL.MASK_FORMER.Q_POSITIONAL_EMBED

        # multi-scale feature projection headss
        ret["in_channels"] = in_channels
        ret["enforce_input_projection"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        # interactive transformer parameters

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        # encoder
        assert cfg.MODEL.MASK_FORMER.ENC_LAYERS >= 1
        ret["enc_layers"] = cfg.MODEL.MASK_FORMER.ENC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        # decoder
        ret["use_decoder"] =  cfg.MODEL.MASK_FORMER.DECODER.USE_DECODER
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DECODER.DEC_LAYERS
        ret["dec_scale_factor"] = cfg.MODEL.MASK_FORMER.DECODER.DEC_SCALE_FACTOR
        # forward prediction head
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        # static background queries
        ret["use_static_bg_queries"] = cfg.MODEL.MASK_FORMER.USE_STATIC_BG_QUERIES
        ret["num_static_bg_queries"] = cfg.MODEL.MASK_FORMER.NUM_STATIC_BG_QUERIES
        
        # rounding
        ret["max_num_rounds"] = cfg.CLICKER.TRAINING.MAX_NUM_REFINEMENT_ROUNDS
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

        assert len(multi_scale_features) == self.num_feature_levels, f"Multi-scale image features ({len(multi_scale_features)} \
            scales) do not have the expected number of feature levels {self.num_feature_levels}"

        ### MEMORY FEATURES & ENCODINGS ###
        # extract multi-scale image features to be used as key and value for Transformer (cross-)attention
        memory, memory_pe = [], []
        size_list = []

        for i in range(self.num_feature_levels):
            # store feature scale
            size_list.append(multi_scale_features[i].shape[-2:])
            
            # positional encoding for current scale
            memory_pe_i = self.pe_layer(multi_scale_features[i], None)                  # T,d',h,w
            memory_pe_i = memory_pe_i.flatten(2).permute(2, 0, 1)                       # (hw),T,d'
            memory_pe_i = self.kv_pos_enc_proj(memory_pe_i)                             # (hw),T,D
            memory_pe.append(memory_pe_i)

            # memory features of current scale
            memory_i = self.input_proj[i](multi_scale_features[i])                      # T,D,h,w
            memory_i = memory_i.flatten(2) + self.level_embed.weight[i][None, :, None]  # T,D,(hw)
            memory.append(memory_i.permute(2, 0, 1))                                    # (hw),T,D
        
        
        if self.training:
            
            # number of corrective iterations
            num_rounds = random.randint(0, self.max_num_rounds)
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

        decoder_output = self.layer_norm(output).transpose(0,1) # Q,T,D -> T,Q,D
        mask_embed = self.mask_embed(decoder_output)            # T,Q,D
        
        # mask prediction
        outputs_mask = torch.einsum("tqd,tdhw->tqhw", mask_embed, mask_features) # T,Q,H,W

        # attention mask
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)   # TxQxhxw
        # boolean attention mask
        attn_mask = (attn_mask.sigmoid() < 0.5)
        
        # TODO: jerry-built
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
        
        # QUERY POSITIONAL ENCODINGS
        query_embed = repeat(self.query_embed, "D -> Q T D", Q=descriptors.shape[1], T=T)   # Q,T,D
        if self.q_positional_encoding:
            pos_coord_embed = get_query_positional_encodings(normalized_clicks.permute(1,0,2),
                                                            self.q_positional_encoding, 
                                                            descriptors.shape[2])           # Q,T,D'
            pos_coord_embed = self.q_pos_enc_proj(pos_coord_embed.to(query_embed.dtype))    # Q,T,D
            query_embed = query_embed + pos_coord_embed                                     # Q,T,D

        # STATIC BACKGROUND QUERIES
        if self.use_static_bg_queries:
            static_bg_queries = repeat(self.static_bg_query, "Bg D -> T Bg D", T=T)         # T,Bg,D
            # append static bg queries to the existing click queries
            descriptors = torch.cat((descriptors, static_bg_queries), dim=1)                # T,Q,D where Q = Q+Bg
            # positional encodings for static bg queries
            static_bg_pe = repeat(self.static_bg_pe, "Bg D -> Bg T D", T=T)
            # append to the existing positional encodings
            query_embed = torch.cat((query_embed, static_bg_pe), dim=0)                     # Q,T,D
            # add bg queries to the count
            num_queries_per_target[-1] += self.num_static_bg_queries
            # add proxy bg clicks to the click
            normalized_clicks = torch.cat([normalized_clicks, torch.full((T, self.num_static_bg_queries, 5), -1.0, device=normalized_clicks.device, dtype=normalized_clicks.dtype)], dim=1)

        # if there's no bg query, remove from record (kinda jerry-built, TODO: improve)
        if num_queries_per_target[-1] == 0:
            num_queries_per_target = num_queries_per_target[:-1]
        
        # MLP
        output = self.queries_nonlinear_projection(descriptors).permute(1,0,2)  # T,Q,D -> Q,T,D
        
        # store predicted mask after each layer, later used in auxiliary loss
        predictions_mask = []
        # pre-encoder prediction and initial QQCA attention mask
        outputs_mask, attn_mask = self.forward_prediction_heads(output, 
                                                                mask_features, 
                                                                attn_mask_target_size=size_list[0],
                                                                orig_clicks=fg_coords+bg_coords)
        predictions_mask.append(outputs_mask)
        
        # ENCODER
        for i in range(self.enc_layers):
            # encoder layers alternate between multi-scale features
            level_index = i % self.num_feature_levels
            
            # unmask completely-masked attention masks
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # IMAGE-QUERY CROSS ATTENTION between queries and image features (intra-frame)
            output = self.encoder.cross_attention_layers[i](tgt=output,                     # Q,T,D
                                                            memory=memory[level_index],     # (hw),T,D
                                                            memory_mask=attn_mask,          # (T*#attn_heads),Q,(hw)
                                                            memory_key_padding_mask=None,
                                                            pos=memory_pe[level_index],
                                                            query_pos=query_embed)
            
            # QUERY-QUERY CROSS ATTENTION between queries (inter-frame)
            #Q,T,D = output.shape
            #qqca_output = self.encoder.query_query_cross_attention_layers[i](
            #    output.view(Q*T,1,D),
            #    tgt_mask=None,
            #    tgt_key_padding_mask=None,
            #    query_pos=query_embed.view((Q*T, 1, D))
            #)
            #output = qqca_output.view(Q,T,D)
            tgt_batched_query, tgt_batched_query_embed, qqca_mask = self.pack_masked_qqca_queries(output, query_embed, num_queries_per_target)
            qqca_output = self.encoder.query_query_cross_attention_layers[i](tgt_batched_query,
                                                                                tgt_mask=None,
                                                                                tgt_key_padding_mask=qqca_mask,
                                                                                query_pos=tgt_batched_query_embed)
            output = self.unpack_masked_qqca_queries(output, qqca_output, num_queries_per_target)
            
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

        # NOTE: the pred masks here are from pre-encoder, and the encoder layers;
        # the decoder output is not in the auxiliary outputs.
        return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

    
    def pack_masked_qqca_queries(
            self, 
            output, 
            query_embed,
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
        target_wise_splits = torch.split(output, num_queries_per_target, dim=0)     # list of T,q_i,D
        target_wise_splits = [part.reshape(-1, D) for part in target_wise_splits]   # list of T*q_i,D
        # do same with p.e.
        target_wise_splits_pe = torch.split(query_embed, num_queries_per_target, dim=0)
        target_wise_splits_pe = [part.reshape(-1, D) for part in target_wise_splits_pe]
        
        # keep a record of how many queries there were for each target
        # this will be used to create an attention mask
        orig_lengths = [r.shape[0] for r in target_wise_splits]
        
        # apply padding for batching
        tgt_batched_query = []
        tgt_batched_query_pe = []
        
        for split_query, split_query_pe in zip(target_wise_splits, target_wise_splits_pe):
            pad_len = max_num_queries - split_query.shape[0]
            if pad_len > 0:
                padding = self.pad_query.expand(pad_len, -1)
                padded_r = torch.cat([split_query, padding], dim=0)

                padding_pe = self.pad_query_pe.expand(pad_len, -1)
                padded_pe = torch.cat([split_query_pe, padding_pe], dim=0)
            else:
                padded_r = split_query
                padded_pe = split_query_pe
            tgt_batched_query.append(padded_r)
            tgt_batched_query_pe.append(padded_pe)

        tgt_batched_query = torch.stack(tgt_batched_query).transpose(0,1)
        tgt_batched_query_pe = torch.stack(tgt_batched_query_pe).transpose(0,1)

        # attention mask
        qqca_mask = torch.arange(max_num_queries).expand(len(orig_lengths), max_num_queries) >= torch.tensor(orig_lengths).unsqueeze(1)
        
        return tgt_batched_query, tgt_batched_query_pe, qqca_mask.to(output.device)

    
    def unpack_masked_qqca_queries(
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