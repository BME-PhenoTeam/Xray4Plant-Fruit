# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor

CUDA_LAUNCH_BLOCKING = 1
USE_JIT = False
relu = nn.ReLU(inplace=True)

from mmseg.registry import MODELS

if USE_JIT:
    _jit = torch.jit.script
else:
    _jit = lambda f: f


@_jit
def gaussian_kernel(kernel_size: int, sigma: float):
    gauss = torch.arange(0, kernel_size) - kernel_size // 2
    gauss = torch.exp(-gauss ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()


@_jit
def gaussian_kernel2d(kernel_size: int, channel: int = 1) -> Tensor:
    '''
    2d gauss kernel, out put shape: [channel, 1, window_size, window_size]
    '''
    k = gaussian_kernel(kernel_size, 1.5)
    k = torch.einsum('i,j->ij', [k, k])
    return k.expand(channel, 1, kernel_size, kernel_size).contiguous()


@_jit
def ssim_index(img1: Tensor,
               img2: Tensor,
               kernel: Tensor,
               nonnegative: bool = True,
               channel_avg: bool = False,
               val_range: float = 1.):
    assert img1.shape == img2.shape
    if len(img1.shape) > 3:
        channel = img1.shape[1]
    else:
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
        channel = 1
    _, channel, height, width = img1.shape
    if img1.dtype == torch.long:
        img1 = img1.float()
    if img2.dtype == torch.long:
        img2 = img2.float()
    L = val_range

    s = 1
    p = 0
    mean1 = F.conv2d(img1, kernel, padding=p, groups=channel, stride=s)
    mean2 = F.conv2d(img2, kernel, padding=p, groups=channel, stride=s)
    mean12 = mean1 * mean2
    mean1 = mean1.pow(2)
    mean2 = mean2.pow(2)

    # https://en.wikipedia.org/wiki/Variance#Definition
    var1 = F.conv2d(img1 ** 2, kernel, padding=p, groups=channel, stride=s) - mean1
    var2 = F.conv2d(img2 ** 2, kernel, padding=p, groups=channel, stride=s) - mean2

    # https://en.wikipedia.org/wiki/Covariance#Definition
    covar = F.conv2d(img1 * img2, kernel, padding=p, groups=channel, stride=s) - mean12

    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2

    # https://en.wikipedia.org/wiki/Structural_similarity#Algorithm
    cs = (2. * covar + c2) / (var1 + var2 + c2)
    # print(covar.mean(), var1.mean(), var2.mean(), cs.mean())  # sparse input could result in large cs
    ss = (2. * mean12 + c1) / (mean1 + mean2 + c1) * cs

    if channel_avg:
        ss, cs = ss.flatten(1), cs.flatten(1)
    else:
        ss, cs = ss.flatten(2), cs.flatten(2)

    ss, cs = ss.mean(dim=-1), cs.mean(dim=-1)
    if nonnegative:
        ss, cs = relu(ss), relu(cs)
    return ss, cs


@_jit
def ms_ssim(
        x: Tensor,
        y: Tensor,
        kernel: Tensor,
        weights: Tensor,
        val_range: float = 1.,
        nonnegative: bool = True
) -> Tensor:
    r"""Returns the MS-SSIM between :math:`x` and :math:`y`.

    modified from https://github.com/francois-rozet/piqa/blob/master/piqa/ssim.py
    """

    css = []
    kernel_size = kernel.shape[-1]
    m = weights.numel()

    for i in range(m):
        if i > 0:
            x = F.avg_pool2d(x, kernel_size=2, ceil_mode=True)
            y = F.avg_pool2d(y, kernel_size=2, ceil_mode=True)
            h, w = x.shape[-2:]
            if h < kernel_size or w < kernel_size:
                weights = weights[:i] / torch.sum(weights[:i])
                break

        ss, cs = ssim_index(
            x, y, kernel,
            channel_avg=False,
            val_range=val_range,
            nonnegative=nonnegative
        )

        css.append(cs if i + 1 < m else ss)

    msss = torch.stack(css, dim=-1) ** weights
    msss = msss.prod(dim=-1).mean(dim=-1)

    return msss


@MODELS.register_module()
class Ms_ssim_loss(nn.Module):
    """
    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        loss_name (str): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_boundary'.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 win_size: int = 11,
                 weights: Tensor = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]),
                 nonnegative: bool = True,
                 process_input: bool = True,
                 loss_name: str = 'loss_ms_ssim'):
        super().__init__()
        self.kernel = gaussian_kernel2d(win_size, 1)
        self.weights = weights
        self.win_size = win_size
        self.nonnegative = nonnegative
        self.process_input = process_input
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
        _, num_classes, h, w = pred_scales.shape
        win_size = min(h, w, self.win_size)
        kernel = self.kernel if win_size == self.win_size else gaussian_kernel2d(win_size, 1)

        kernel = kernel.to(pred_scales.dtype).to(pred_scales.device)
        weights = self.weights.to(pred_scales.dtype).to(pred_scales.device)
        # if kernel.device != pred.device:
        #     kernel.to(pred.device)

        if self.process_input:
            pred = F.softmax(pred_scales, dim=1)
            target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        loss = 0.
        for i in range(num_classes):
            ss = ms_ssim(pred_scales[:, [i]], target[:, [i]], kernel, weights, nonnegative=self.nonnegative)
            loss += 1. - ss.mean()
        return loss / num_classes

    @property
    def loss_name(self):
        return self.loss_name_
