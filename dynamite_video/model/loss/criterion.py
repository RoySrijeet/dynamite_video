"""
DynaMITe-Video criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import ( # type: ignore
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

import dynamite_video.utils.distributed as dist_utils
from dynamite_video.model.loss.loss_functions import sigmoid_ce_loss_jit, dice_loss_jit, calculate_uncertainty


class SetFinalCriterion(nn.Module):
    """
    Meta
    """

    def __init__(self, weight_dict, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

        # PointRend (pointwise mask loss) parameters
        self.num_points = num_points                             # N in PointRend paper
        self.oversample_ratio = oversample_ratio                 # k in PointRend paper
        self.importance_sample_ratio = importance_sample_ratio   # beta in PointRend paper
    

    def loss_masks(self, outputs, targets, num_masks, num_queries_per_object):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        pred_masks = outputs["pred_masks"] # T,Q,H,W

        # gt semantic masks, T,H,W
        target_masks = torch.stack([t["semantic_masks"] for t in targets]).to(dtype=torch.float32)
        target_masks = target_masks[:, None]    # T,1,H,W
        # ignore masks, T,H,W
        ignore_masks = torch.stack([t["ignore_masks"] for t in targets])
        ignore_masks = ignore_masks[:, None]    # T,1,H,W

        # resize ignore mask to pred_masks resolution (1/4)
        ignore_masks_ds = F.interpolate(
            ignore_masks.type_as(pred_masks), scale_factor=0.25, mode='bilinear', align_corners=False
        )
        ignore_masks_ds = (ignore_masks_ds > 0.5).type_as(pred_masks).detach()
        assert ignore_masks_ds.shape[-2:] == pred_masks.shape[-2:], f"Shape mismatch: {ignore_masks.shape}, {pred_masks.shape}"

        with torch.no_grad():
            concat_ignore_mask_logits = torch.cat((ignore_masks_ds, pred_masks), 1)
            # sample point_coords (PointRend) - [T, P, 2]
            point_coords = get_uncertain_point_coords_with_randomness(concat_ignore_mask_logits,
                                                                    lambda logits: calculate_uncertainty(logits),
                                                                    self.num_points,
                                                                    self.oversample_ratio,
                                                                    self.importance_sample_ratio,
                                                                )
            # get gt labels
            point_labels = point_sample(target_masks.float(), point_coords, mode='nearest', align_corners=False).long().squeeze(1)  # [T, P]
            point_ignore = point_sample(ignore_masks.float(), point_coords, mode='nearest', align_corners=False).bool().squeeze(1)  # [T, P]

        point_logits = point_sample(pred_masks, point_coords, align_corners=False)                  # T,Q,P
        point_labels = torch.where(point_ignore, torch.full_like(point_labels, -100), point_labels) # T,P

        loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels)

        losses = {
            "loss_mask": loss_mask,
            "loss_dice": torch.tensor([0.]).to(loss_mask.device), #dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del pred_masks
        del target_masks
        return losses

    def get_loss(self, loss, outputs, targets, num_masks, num_queries_per_object):
        loss_map = {
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, num_masks, num_queries_per_object)

    def forward(self, outputs, targets, instance_ids, num_queries_per_object):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # for i,t in enumerate(targets):
        #     target_bg_mask = t['bg_mask']
        #     targets[i]["masks"] = torch.cat((t["masks"], target_bg_mask.unsqueeze(0)), dim=0)   
        
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num of masks to be predicted = (num of instances in each frame + BG) * num of frames
        num_masks = len(instance_ids) * len(targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=outputs['pred_masks'].device)
        if dist_utils.is_distributed():
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
