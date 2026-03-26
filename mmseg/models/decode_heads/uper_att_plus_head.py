import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM


# 定义论文中的 Patch-Enhanced Attention 模块
class PatchEnhancedAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(PatchEnhancedAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平方向池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直方向池化
        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 水平注意力
        identity = x
        h = self.pool_h(x).permute(0, 1, 3, 2)  # (N, C, W, H)
        h = self.conv1(h).relu()
        h = self.conv2(h).sigmoid().permute(0, 1, 3, 2)  # 恢复形状

        # 垂直注意力
        w = self.pool_w(x)
        w = self.conv1(w).relu()
        w = self.conv2(w).sigmoid()

        # 输出结合水平和垂直注意力
        out = identity * h * w
        return out


# 多任务输出头，用于语义分割和边缘检测
class MultiTaskHead(nn.Module):
    def __init__(self, in_channels, seg_channels, edge_channels):
        super(MultiTaskHead, self).__init__()
        # 语义分割头
        self.segmentation_head = nn.Conv2d(in_channels, seg_channels, kernel_size=1)
        # 边缘检测头
        self.edge_detection_head = nn.Conv2d(in_channels, edge_channels, kernel_size=1)

    def forward(self, x):
        seg_output = self.segmentation_head(x)
        edge_output = self.edge_detection_head(x)
        return seg_output, edge_output


@MODELS.register_module()
class UPerATTPLUSHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding with multi-task learning (Segmentation and Edge Detection)."""

    def __init__(self, pool_scales=(1, 2, 3, 6), seg_channels=19, edge_channels=1, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

        # Bottleneck with standard convolution
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        # FPN Module with depthwise separable convolutions
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.pea_layers = nn.ModuleList()  # 添加 PEA 模块
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = DepthwiseSeparableConvModule(
                in_channels,
                self.channels,
                kernel_size=1,  # 注意：1x1卷积可以直接使用普通卷积
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)

            fpn_conv = DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.pea_layers.append(PatchEnhancedAttention(self.channels))  # 为每层FPN添加PEA

        self.fpn_bottleneck = DepthwiseSeparableConvModule(
                len(self.in_channels) * self.channels,
                self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
        )

        # 初始化多任务头
        self.multi_task_head = MultiTaskHead(self.channels, seg_channels, edge_channels)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)  # 使用标准卷积的 bottleneck

        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.pea_layers[i](self.fpn_convs[i](laterals[i]))  # 在 FPN 输出上应用 PEA
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs, gt_seg=None):
        """Forward function."""
        output = self._forward_feature(inputs)

        # 获取多任务输出
        seg_output, edge_output = self.multi_task_head(output)

        if self.training:
            # 确保提供语义分割标签
            assert gt_seg is not None, "Ground truth for segmentation is required during training"

            # 交叉熵损失（语义分割）
            loss_seg = nn.CrossEntropyLoss()(seg_output, gt_seg)

            # 边界损失（基于预测边界的梯度）
            seg_grad = torch.gradient(seg_output.softmax(dim=1), dim=(-2, -1))  # 分割预测的梯度
            edge_pred = seg_grad[0].abs() + seg_grad[1].abs()  # 合并水平和垂直梯度
            edge_gt = (torch.gradient(gt_seg.float(), dim=(-2, -1))[0].abs() +
                       torch.gradient(gt_seg.float(), dim=(-2, -1))[1].abs())  # GT 的边界

            # 边界损失（使用二值交叉熵）
            loss_edge = nn.BCEWithLogitsLoss()(edge_pred, edge_gt)

            # 总损失
            lambda_seg = 1.0  # 语义分割损失的权重
            lambda_edge = 0.1  # 边界损失的权重
            total_loss = lambda_seg * loss_seg + lambda_edge * loss_edge

            return {
                "loss_seg": loss_seg,
                "loss_edge": loss_edge,
                "total_loss": total_loss
            }
        else:
            # 推理模式：仅返回语义分割输出
            return {"seg_output": seg_output}
            # loss = seg_output+edge_output
            # return loss