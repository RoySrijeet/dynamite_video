import torch
import torch.nn.functional as F

from typing import Optional

def calculate_uncertainty(logits):
    """
    # TODO - update description
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    # assert logits.shape[1] == 1
    # gt_class_logits = logits.clone()
    # return -(torch.abs(gt_class_logits))
    # logits: [N, 1+C, ...]. 1st channel for dim1 is the ignore mask
    ignore_mask, logits = logits.split((1, logits.size(1) - 1), 1)
    ignore_mask = ignore_mask.bool()

    top2 = logits.topk(k=2, dim=1, largest=True, sorted=True).values
    uncertainty = top2[:, 1] - top2[:, 0]  # [N, ...]
    uncertainty = uncertainty.unsqueeze(1)  # [N, 1, ....]

    # assign very low uncertainty to points which are supposed to be ignored
    uncertainty = torch.where(ignore_mask, torch.full_like(uncertainty, -1e3), uncertainty)
    return uncertainty

def _reduce(x, reduction, mean_factor=None):
    if mean_factor is None:
        mean_factor = x.numel()

    if reduction == 'none':
        return x
    elif reduction == 'sum':
        return x.sum()
    elif reduction == 'mean':
        return x.sum() / mean_factor
    else:
        raise ValueError(f"Invalid reduction argument: '{reduction}'")

def dice_loss_tarvis(input: torch.Tensor, target: torch.Tensor, ignore_mask: Optional[torch.Tensor] = None, eps: float = 1e-6,
              reduction: Optional[str] = "mean"):
    """
    Computes the DICE or soft IoU loss.
    :param input: tensor of shape [N, *]
    :param target: tensor with shape identical to input
    :param ignore_mask: tensor of same shape as input. non-zero values in this mask will be
    :param eps
    :param reduction: type of reduction over the first dimension
    excluded from the loss calculation.
    :return: tensor
    """
    assert input.shape == target.shape, "Shape mismatch between input ({}) and target ({})".format(input.shape, target.shape)
    assert input.dtype == target.dtype

    if torch.is_tensor(ignore_mask):
        assert ignore_mask.dtype == torch.bool
        assert input.shape == ignore_mask.shape, f"Shape mismatch between input ({input.shape}) and ignore mask ({ignore_mask.shape})"
        input = torch.where(ignore_mask, torch.zeros_like(input), input)
        target = torch.where(ignore_mask, torch.zeros_like(target), target)

    input = input.flatten(1)
    target = target.detach().flatten(1)

    numerator = 2.0 * (input * target).mean(1)
    denominator = (input + target).mean(1)

    soft_iou = (numerator + eps) / (denominator + eps)

    loss = torch.where(numerator > eps, 1. - soft_iou, soft_iou * 0.)
    return _reduce(loss, reduction)


def multiclass_dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-6,
                         check_target_validity: bool = True, ignore_zero_class: bool = True,
                         ignore_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes DICE loss for multi-class predictions. API inputs are identical to torch.nn.functional.cross_entropy()
    :param input: tensor of shape [N, C, *] with unscaled logits
    :param target: tensor of shape [N, *]
    :param eps:
    :param check_target_validity: checks if the values in the target are valid
    :param ignore_zero_class: Ignore the IoU for class ID 0
    :param ignore_mask: optional tensor of shape [N, *]
    :return: tensor
    """
    assert input.ndim >= 2
    input = input.softmax(1)
    num_classes = input.size(1)

    if check_target_validity:
        class_ids = target.unique()
        assert not torch.any(torch.logical_or(class_ids < 0, class_ids >= num_classes)), \
            f"Number of classes = {num_classes}, but target has the following class IDs: {class_ids.tolist()}"

    target = torch.stack([target == cls_id for cls_id in range(0, num_classes)], 1).to(dtype=input.dtype)  # [N, C, *]

    if ignore_zero_class:
        input = input[:, 1:]
        target = target[:, 1:]

    if ignore_mask is not None:
        ignore_mask = ignore_mask.unsqueeze(1)
        expand_dims = [-1, input.size(1)] + ([-1] * (ignore_mask.ndim - 2))
        ignore_mask = ignore_mask.expand(*expand_dims)

    return dice_loss_tarvis(input, target, eps=eps, ignore_mask=ignore_mask)



def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        input: torch.Tensor,
        target: torch.Tensor,
    ):
    """
    Args:
        input: mask prediction logits, expected shape: T, Q, P
        target: g.t. multi-class classification label for each element in pred_logits,
                expected shape: T, P
    Returns:
        Loss tensor
    """
    loss = F.cross_entropy(input, target, ignore_index=-100)
    return loss


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule
