# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS


@MODELS.register_module()
class Focal_loss(nn.Module):
    """
    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 alpha=0.5,
                 gamma=2.0,
                 size_average=True,
                 ignore_index=255,
                 loss_name: str = 'loss_focal'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name

    def forward(self, pred_scales, target):
        """Forward function.
        Args:
            pred_scales (Tensor): Predictions of the boundary head.
            target (Tensor): Ground truth of the boundary.

        Returns:
            Tensor: Loss tensor.
        """
        ce_loss = F.cross_entropy(
            pred_scales, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return self.loss_weight * focal_loss.mean()
        else:
            return self.loss_weight * focal_loss.sum()

    @property
    def loss_name(self):
        return self.loss_name_
