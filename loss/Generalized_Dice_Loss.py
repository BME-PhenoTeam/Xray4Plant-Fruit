# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS

"""compute the weighted dice_loss
Args:
    pred (tensor): prediction after softmax, shape(bath_size, channels, height, width)
    target (tensor): gt, shape(bath_size, channels, height, width)
Returns:
    gldice_loss: loss value
"""


@MODELS.register_module()
class Generalized_Dice_Loss(nn.Module):
    """
    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 process_input=True,
                 loss_name: str = 'loss_GD'):
        super().__init__()
        self.process_input = process_input
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name

    def forward(self, pred_scales, target, epsilon=1e-6):
        """Forward function.
        Args:
            epsilon:
            pred_scales (Tensor): Predictions of the boundary head.
            target (Tensor): Ground truth of the boundary.

        Returns:
            Tensor: Loss tensor.
        """
        num_classes = pred_scales.shape[1]

        if self.process_input:
            pred_scales = F.softmax(pred_scales, dim=1)
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        wei = torch.sum(target, axis=[0, 2, 3])  # (n_class,)
        wei = 1 / (wei ** 2 + epsilon)
        intersection = torch.sum(wei * torch.sum(pred_scales * target, axis=[0, 2, 3]))
        union = torch.sum(wei * torch.sum(pred_scales + target, axis=[0, 2, 3]))
        gldice_loss = 1 - (2. * intersection) / (union + epsilon)

        return gldice_loss * self.loss_weight

    @property
    def loss_name(self):
        return self.loss_name_
