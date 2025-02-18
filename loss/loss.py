# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union, Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from collections import ChainMap

from mmseg.models.utils.weight_init import weight_init
from mmseg.models.losses.unet3_plus_ms_ssim_loss import Ms_ssim_loss
from mmseg.models.losses.unet_plus_iou_loss import IOU_loss
from mmseg.models.losses.unet3_plus_generalized_dice_loss import Generalized_Dice_Loss
from mmseg.models.losses.unet3_plus_focal_Loss import Focal_loss
from mmseg.models.losses import accuracy
from mmseg.models.losses.boundary_loss import BoundaryLoss
from mmseg.models.losses.dice_loss import DiceLoss

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList

from mmseg.visualization import SegLocalVisualizer


@MODELS.register_module()
class LOSS(BaseDecodeHead):
    def __init__(self,
                 loss_type='generalized_dice_focal_boundary_ms_ssim',
                 process_input=True,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        """----------------------------------------------loss选择-----------------------------------------------------"""
        """
        self.loss_decode[0] ---> FocalLoss
        self.loss_decode[1] ---> Generalized_Dice_Loss
        self.loss_decode[2] ---> MS_SSIMLoss
        self.loss_decode[3] ---> BoundaryLoss
        self.loss_decode[4] ---> DiceLoss
        """
        self.loss_type = loss_type
        self.process_input = process_input

        if loss_type == 'dice_focal':
            self.focal_loss = Focal_loss(ignore_index=255, size_average=True, loss_weight=1.0)
            self.DiceLoss = DiceLoss(loss_weight=1.0)

        elif loss_type == 'focal':
            self.focal_loss = Focal_loss(ignore_index=255, size_average=True, loss_weight=1.0)

        elif loss_type == 'dice_focal_boundary':
            self.focal_loss = Focal_loss(ignore_index=255, size_average=True, loss_weight=1.0)
            self.DiceLoss = DiceLoss(loss_weight=1.0)
            self.BoundaryLoss = BoundaryLoss(loss_weight=20)

        elif loss_type == 'dice_focal_boundary_ms_ssim':
            self.focal_loss = Focal_loss(ignore_index=255, size_average=True, loss_weight=1.0)
            self.DiceLoss = DiceLoss(loss_weight=1.0)
            self.BoundaryLoss = BoundaryLoss(loss_weight=20)
            self.ms_ssim_loss = Ms_ssim_loss(process_input=not process_input)

        elif loss_type == 'generalized_dice_focal':
            self.focal_loss = Focal_loss(ignore_index=255, size_average=True, loss_weight=1.0)
            self.Generalized_Dice_Loss = Generalized_Dice_Loss(process_input=not process_input)

        elif loss_type == 'generalized_dice_focal_boundary':
            self.focal_loss = Focal_loss(ignore_index=255, size_average=True, loss_weight=1.0)
            self.Generalized_Dice_Loss = Generalized_Dice_Loss(process_input=not process_input)
            self.BoundaryLoss = BoundaryLoss(loss_weight=20)

        elif loss_type == 'generalized_dice_focal_boundary_ms_ssim':
            self.focal_loss = Focal_loss(ignore_index=255, size_average=True, loss_weight=1.0)
            self.Generalized_Dice_Loss = Generalized_Dice_Loss(process_input=not process_input)
            self.BoundaryLoss = BoundaryLoss(loss_weight=20)
            self.ms_ssim_loss = Ms_ssim_loss(process_input=not process_input)
        
        elif loss_type == 'generalized_dice_focal_ms_ssim':
            self.focal_loss = Focal_loss(ignore_index=255, size_average=True, loss_weight=1.0)
            self.Generalized_Dice_Loss = Generalized_Dice_Loss(process_input=not process_input)   
            self.ms_ssim_loss = Ms_ssim_loss(process_input=not process_input)
            
        else:
            raise ValueError(f'Unknown loss type: {loss_type}')

    """--------------------------------------------------工具---------------------------------------------------------"""
    def resize(self, x, h, w) -> torch.Tensor:
        _, _, xh, xw = x.shape
        if xh != h or xw != w:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

    def onehot_softmax(self, pred, target: torch.Tensor, process_target=True):
        _, num_classes, h, w = pred.shape
        pred = F.softmax(pred, dim=1)

        if process_target:
            target = torch.clamp(target, 0, num_classes)
            target = F.one_hot(target, num_classes=num_classes + 1)[..., :num_classes].permute(0, 3, 1, 2).contiguous().to(pred.dtype)
        return pred, target

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
        gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]  # 掩膜读取
        gt_edge_segs = [data_sample.gt_edge_map.data for data_sample in batch_data_samples]  # 边缘标签读取

        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)

        return gt_sem_segs, gt_edge_segs  # 得到掩膜

    """-----------------------------------------------兼容各LOSS------------------------------------------------------"""
    def _forward_loss(self, preds, targets, loss_choose, aux_choose, loss_name):
        loss_dict = {}

        if loss_name == 'focal' or loss_name == 'dice':
            if loss_name == 'focal':
                loss_dict['acc_seg'] = accuracy(preds['final_pred'], targets, ignore_index=255)       # acc评估,只评估一次
            loss = loss_choose(preds['final_pred'], targets)                                          # focal内置F.cross_entropy自动进行softmax
        elif loss_name == 'boundary':
            loss = loss_choose(preds['fin_cls_seg'], targets)
        elif loss_name == 'generalized_dice' or loss_name == 'msssim':                                # 手动softmax，loss里也兼容，需要关闭process_input
            final_pred, targets = self.onehot_softmax(preds['final_pred'], targets)
            loss = loss_choose(final_pred, targets)

        if aux_choose:
            num_aux, aux_loss = 0, 0.
            for key in preds:
                if 'aux' in key:
                    num_aux += 1
                    if loss_name == 'generalized_dice' or loss_name == 'msssim':
                        preds[key], targets = self.onehot_softmax(preds[key], targets, process_target=False)
                    aux_loss += loss_choose(preds[key], targets)    # aux_loss
            if num_aux > 0:
                aux_loss = aux_loss / num_aux * self.aux_weight     # aux_loss求平均
                loss += aux_loss
                loss_dict[f'{loss_name}_total_loss'] = loss         # total_loss = head_loss + aux_loss
        else:
            loss_dict[f'{loss_name}_loss'] = loss

        return loss_dict

    """---------------------------------------选择loss并参与计算--------------------------------------------------------"""
    def loss_by_feat(self, seg_logits: Tuple[Tensor], batch_data_samples: SampleList) -> Tuple[Any, Dict[str, Any]]:

        preds = seg_logits
        target, bd_label = self._stack_batch_gt(batch_data_samples)  # 得到掩膜 (B, 1, W, H), 得到边缘标签

        targets = target.squeeze(1)  # (B, 1, W, H)--->(B, W, H)
        bd_label = bd_label.squeeze(1)  # (B, 1, W, H)--->(B, W, H)

        if not isinstance(preds, dict):
            preds = {'final_pred': preds}

        if self.loss_type == 'dice_focal':
            focal_loss = self._forward_loss(preds, targets, self.focal_loss, True, loss_name='focal')
            dice_loss = self._forward_loss(preds, targets, self.DiceLoss, False, loss_name='dice')
            dice_focal_loss = dict(ChainMap(focal_loss, dice_loss))         # 合并字典

            return dice_focal_loss
        
        elif self.loss_type == 'focal':
            focal_loss = self._forward_loss(preds, targets, self.focal_loss, False, loss_name='focal')

            return focal_loss        

        elif self.loss_type == 'dice_focal_boundary':
            focal_loss = self._forward_loss(preds, targets, self.focal_loss, True, loss_name='focal')
            dice_loss = self._forward_loss(preds, targets, self.DiceLoss, False, loss_name='dice')
            boundar_loss = self._forward_loss(preds, bd_label, self.BoundaryLoss, False, loss_name='boundary')
            dice_focal_loss = dict(ChainMap(focal_loss, dice_loss, boundar_loss))         # 合并字典

            return dice_focal_loss

        elif self.loss_type == 'dice_focal_boundary_ms_ssim':
            focal_loss = self._forward_loss(preds, targets, self.focal_loss, True, loss_name='focal')
            dice_loss = self._forward_loss(preds, targets, self.DiceLoss, False, loss_name='dice')
            boundar_loss = self._forward_loss(preds, bd_label, self.BoundaryLoss, False, loss_name='boundary')
            ms_ssim_loss = self._forward_loss(preds, targets, self.ms_ssim_loss, False, loss_name='msssim')
            dice_focal_loss = dict(ChainMap(focal_loss, dice_loss, boundar_loss, ms_ssim_loss))         # 合并字典

            return dice_focal_loss

        elif self.loss_type == 'generalized_dice_focal':
            focal_loss = self._forward_loss(preds, targets, self.focal_loss, True, loss_name='focal')
            generalized_dice_loss = self._forward_loss(preds, targets, self.Generalized_Dice_Loss, False, loss_name='generalized_dice')
            dice_focal_loss = dict(ChainMap(focal_loss, generalized_dice_loss))         # 合并字典

            return dice_focal_loss

        elif self.loss_type == 'generalized_dice_focal_boundary':
            focal_loss = self._forward_loss(preds, targets, self.focal_loss, True, loss_name='focal')
            generalized_dice_loss = self._forward_loss(preds, targets, self.Generalized_Dice_Loss, False, loss_name='generalized_dice')
            boundar_loss = self._forward_loss(preds, bd_label, self.BoundaryLoss, False, loss_name='boundary')
            dice_focal_loss = dict(ChainMap(focal_loss, generalized_dice_loss, boundar_loss))         # 合并字典

            return dice_focal_loss

        elif self.loss_type == 'generalized_dice_focal_boundary_ms_ssim':
            focal_loss = self._forward_loss(preds, targets, self.focal_loss, True, loss_name='focal')
            generalized_dice_loss = self._forward_loss(preds, targets, self.Generalized_Dice_Loss, False, loss_name='generalized_dice')
            boundar_loss = self._forward_loss(preds, bd_label, self.BoundaryLoss, False, loss_name='boundary')
            ms_ssim_loss = self._forward_loss(preds, targets, self.ms_ssim_loss, False, loss_name='msssim')
            dice_focal_loss = dict(ChainMap(focal_loss, generalized_dice_loss, boundar_loss, ms_ssim_loss))         # 合并字典

            return dice_focal_loss
        
        elif self.loss_type == 'generalized_dice_focal_ms_ssim':
            focal_loss = self._forward_loss(preds, targets, self.focal_loss, True, loss_name='focal')
            generalized_dice_loss = self._forward_loss(preds, targets, self.Generalized_Dice_Loss, False, loss_name='generalized_dice')
            ms_ssim_loss = self._forward_loss(preds, targets, self.ms_ssim_loss, False, loss_name='msssim')
            dice_focal_loss = dict(ChainMap(focal_loss, generalized_dice_loss, ms_ssim_loss))         # 合并字典

            return dice_focal_loss        

