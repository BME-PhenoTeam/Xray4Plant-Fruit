from mmseg.registry import MODELS

import torch
import torch.nn as nn
from mmseg.models.backbones.replknet import RepLKNet
from mmseg.models.utils.weight_init import init_weights

from mmseg.models.attention.msca import MSCA
from mmseg.models.attention.simam import SimAM

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


@MODELS.register_module()
class UNet3Plus(nn.Module):
    def __init__(self,
                 in_channels=1,
                 feature_scale=4,
                 filters=None,
                 is_deconv=True,
                 is_batchnorm=True,
                 Attention_choose=False,
                 Original_encoder=True):

        super(UNet3Plus, self).__init__()
        if filters is None:
            filters = [64, 64, 128, 256, 512]
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.Attention_choose = Attention_choose
        self.Original_encoder = Original_encoder

        # -------------Encoder--------------
        if self.Original_encoder:
            self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
            self.maxpool1 = nn.MaxPool2d(kernel_size=2)

            self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2)

            self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
            self.maxpool3 = nn.MaxPool2d(kernel_size=2)

            self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
            self.maxpool4 = nn.MaxPool2d(kernel_size=2)

            self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        else:
            self.encoder = RepLKNet(large_kernel_sizes=[31, 29, 27, 13],
                                    layers=[2, 2, 18, 2],
                                    channels=[64, 128, 256, 512],
                                    drop_path_rate=0.5,
                                    small_kernel=5,
                                    dw_ratio=1,
                                    num_classes=None,
                                    out_indices=(0, 1, 2, 3),
                                    use_checkpoint=False,
                                    small_kernel_merged=False,
                                    use_sync_bn=True,
                                    pretrained=None,)        # 使用RepLKNet作为编码器

        # -------------MSCA_Attention-------------
        if self.Attention_choose:
            self.MSCA = MSCA(dim=1024)

        # -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 4

        self.UpChannels = self.CatChannels * self.CatBlocks
        self.Decoder_cat = self.CatChannels * 4     # 拼接使用

        '''stage 4d'''
        if self.Original_encoder:
            self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
            self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
            self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
            self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        self.conv4d_1 = nn.Conv2d(self.UpChannels if self.Original_encoder else self.Decoder_cat, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        # -------------SIMAM_Attention-------------
        if self.Attention_choose:
            self.SimAM_4d = SimAM()

        # h1
        if self.Original_encoder:
            self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
            self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
            self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
            self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        self.conv3d_1 = nn.Conv2d(self.UpChannels if self.Original_encoder else self.Decoder_cat, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        # -------------SIMAM_Attention-------------
        if self.Attention_choose:
            self.SimAM_3d = SimAM()

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        if self.Original_encoder:
            self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
            self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
            self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
            self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv2d(self.UpChannels if self.Original_encoder else self.Decoder_cat, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        # -------------SIMAM_Attention-------------
        if self.Attention_choose:
            self.SimAM_2d = SimAM()

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        if self.Original_encoder:
            self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
            self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
            self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        self.conv1d_1 = nn.Conv2d(self.UpChannels if self.Original_encoder else self.Decoder_cat, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------SIMAM_Attention-------------
        if self.Attention_choose:
            self.SimAM_1d = SimAM()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # -------------Encoder-------------
        if self.Original_encoder:
            h1 = self.conv1(inputs)  # h1->320*320*64

            h2 = self.maxpool1(h1)
            h2 = self.conv2(h2)  # h2->160*160*128

            h3 = self.maxpool2(h2)
            h3 = self.conv3(h3)  # h3->80*80*256

            h4 = self.maxpool3(h3)
            h4 = self.conv4(h4)  # h4->40*40*512

            h5 = self.maxpool4(h4)
            hd5 = self.conv5(h5)  # h5->20*20*1024
        else:
            enc_outs = []
            level_outputs = self.encoder(inputs)        # 替换为ReplkNet作为编码器特征提取
            for level_out in level_outputs:
                enc_outs.append(level_out)
            h2, h3, h4, hd5 = enc_outs

        # -------------MSCA_Attention-------------
        if self.Attention_choose:
            hd5 = self.MSCA(hd5)

        # -------------Decoder-------------
        if self.Original_encoder:
            h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        if self.Original_encoder:
            hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  
        else:
            hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  
            if self.Attention_choose:
                hd4 = self.SimAM_4d(hd4)

        if self.Original_encoder:
            h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        if self.Original_encoder:
            hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  
        else:
            hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) 
            if self.Attention_choose:
                hd3 = self.SimAM_3d(hd3)

        if self.Original_encoder:
            h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        if self.Original_encoder:
            hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) 
        else:
            hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  
            if self.Attention_choose:
                hd2 = self.SimAM_3d(hd2)

        if self.Original_encoder:
            h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        if self.Original_encoder:
            hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  
        else:
            hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  
            if self.Attention_choose:
                hd1 = self.SimAM_3d(hd1)

        return hd5, hd4, hd3, hd2, hd1




