import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import ( # type: ignore
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from dynamite_video.utils.misc import is_dist_avail_and_initialized
from dynamite_video.model.loss.loss_functions import dice_loss_jit, sigmoid_ce_loss_jit, calculate_uncertainty


class SetFinalCriterion(nn.Module):
    """
    DynaMITe-Video loss calculation
    """

    def __init__(
            self, 
            weight_dict, 
            losses, 
            num_points, 
            oversample_ratio, 
            importance_sample_ratio
    ):
        """Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight
            losses: list of all the losses to be applied. See get_loss for list of available losses
            num_points: N, in PointRend paper
            oversample_ratio: k in PointRend paper
            importance_sample_ratio: beta in PointRend paper
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
    

    def loss_masks(
            self, 
            outputs, 
            targets,
            num_masks, 
            num_queries_per_object
    ):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Args:
            outputs: dict with key "pred_masks" containing prediction logits
            targets: list of dicts, one for each frame, with key "binary_masks"
                    containing g.t. masks
        """
        assert "pred_masks" in outputs

        # ground truth binary masks
        gt_masks = [t["binary_masks"] for t in targets]
        gt_masks = torch.cat(gt_masks, dim=0).to(dtype=torch.float16).unsqueeze(1)  # T*N,1,H,W

        # Accumulate mask for each object (as there might be multiple clicks per object) and background
        new_outputs = []
        for _, mask_pred in enumerate(outputs['pred_masks']):
            H,W = mask_pred.shape[1:]
            temp_out = []
            splited_masks = torch.split(mask_pred, num_queries_per_object, dim=0)
            for m in splited_masks:
                temp_out.append(torch.max(m, dim=0).values)
            new_outputs.append(torch.stack(temp_out))
            # new_outputs.append(torch.stack(temp_out[:-1]))  # excluding bg
        
        # predicted masks at 1/4th resolution
        pred_masks = torch.cat(new_outputs, dim=0).unsqueeze(1)    # T*N,1,h,w

        with torch.no_grad():
            
            # sample PointRend points from predicted masks. Behind the scenes:
            # 1. Sample P points from the input [T,(N+1),h,w] mask logits. This produces a (T,P,2) tensor, 
            # where P = num_points * oversample_ratio
            # 2. Find the predicted logits at these locations. This produces a [T,(N+1),P] tensor of pred logits
            # 3. Compute uncertainty of the sampled predicted logits. Uncertainty is measured by the L1 distance 
            # between 0.0 and the logit (if raw prediction is 0.75, uncertainty = -0.75)
            # 4. Out of the P points, most uncertain (ß * p) points are considered, where ß = importance sampling 
            # ratio and p = num_points
            # 5. (1 - ß) * p points are randomly sampled additionally
            point_coords = get_uncertain_point_coords_with_randomness(pred_masks,
                                                                    lambda logits: calculate_uncertainty(logits),
                                                                    self.num_points,
                                                                    self.oversample_ratio,
                                                                    self.importance_sample_ratio,
                                                                )   # T,num_points,2
            # get gt labels at the sampled locations
            point_labels = point_sample(gt_masks, point_coords, align_corners=False).squeeze(1)    # T*N,num_points
        
        point_logits = point_sample(pred_masks, point_coords, align_corners=False).squeeze(1)      # T*N,num_points

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del gt_masks, pred_masks
        return losses

    
    def get_loss(
            self,
            loss, 
            outputs, 
            targets, 
            num_masks, 
            num_queries_per_object
    ):
        loss_map = {
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, num_masks, num_queries_per_object)

    
    def forward(
            self, 
            outputs, 
            targets,
            num_queries_per_object
    ):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors with keys:
                    "pred_outputs": final prediction logits
                    "aux_outputs": prediction logits from transformer intermediate layers
             targets: list of dicts with g.t. masks, one entry per frame of clip
             num_queries_per_object: num of queries per object, np.ndarray [T, N]
        """
        # Stack object binary masks and background masks and 
        # Compute number of target boxes accross all nodes for normalization purposes
        num_masks = 0
        for i,t in enumerate(targets):
            target_bg_mask = t['bg_masks']
            if target_bg_mask is not None:
                targets[i]["binary_masks"] = torch.cat((t["binary_masks"], target_bg_mask.unsqueeze(0)), dim=0)
            num_masks += len(targets[i]["binary_masks"])
        
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=outputs['pred_masks'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, num_masks, num_queries_per_object))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_masks, num_queries_per_object)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    
    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

