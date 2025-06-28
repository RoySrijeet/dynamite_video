import torch
import torch.nn.functional as F

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of shape T,N,P with prediction logits
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    T,N,P = inputs.shape 
    inputs = inputs.reshape((T*N),P)
    targets = targets.reshape((T*N),P)
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of shape T,N,P with prediction logits
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")    # T,N,P
    # mean loss over all points in each binary mask
    # sum over all masks in each frame
    # sum over all frames
    # normalized by total num of frames
    ret = loss.mean(2).sum(1).sum() / num_masks
    return ret


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    Uncertainty is estimates as the L1 distance between 0. and the logit prediction.
    Points sampled on the ignore masks are used to filter out locations to be ignored 
    and are assigned very low uncertainty (-1000)

    Example - if predicted logit is 0.75, uncertainty = -0.75

    In `get_uncertain_point_coords_with_randomness()`, the most uncertain points can be
    sampled by picking the top-K uncertainties (-0.01 > -0.75). Any ignored point is never
    considered (-1000 < -x, where x is any pred logit in (0,1)).

    Args:
        logits: tensor of shape T,(N+1),H,W where the last channel in second dimension is
                the ignore mask
    """
    logits, ignore_mask = logits.split((logits.size(1)-1, 1), 1)    # T,N,P & T,1,P
    ignore_mask = ignore_mask.bool()                                # T,1,P

    gt_class_logits = logits.clone()
    uncertainty = -(torch.abs(gt_class_logits))                     # T,1,P

    # ignore regions get low uncertainty
    uncertainty = torch.where(ignore_mask, torch.full_like(uncertainty, -1e3), uncertainty)
    return uncertainty