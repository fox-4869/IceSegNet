import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmseg.registry import MODELS


# 定义SPIEM模块
class SPIEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPIEM, self).__init__()
        self.position_pooling = nn.AdaptiveAvgPool2d(1)  # 位置池化
        self.saliency_pooling = nn.AdaptiveMaxPool2d(1)  # 显著性池化
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        pos_info = self.position_pooling(x)
        sal_info = self.saliency_pooling(x)
        combined_info = pos_info + sal_info
        return self.conv1x1(combined_info)


# 定义LR-FPN中的CIM模块
class CIM(nn.Module):
    def __init__(self, in_channels):
        super(CIM, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.dilated_depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels)
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        local_spatial_info = self.depthwise_conv(x)
        global_spatial_info = self.dilated_depthwise_conv(x)
        channel_weights = self.channel_interaction(x)
        return (local_spatial_info + global_spatial_info) * channel_weights


# 修改SEFPN类，集成SPIEM和CIM
@MODELS.register_module()
class ACCFPN(BaseModule):
    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, 
                 add_extra_convs=False, extra_convs_on_inputs=True, norm_cfg=None, act_cfg=None):
        super(ACCFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = self.num_ins if end_level == -1 else end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        # 初始化SPIEM模块
        self.spiem = SPIEM(in_channels[0], out_channels)  # 使用低层特征
        self.cims = nn.ModuleList([CIM(out_channels) for _ in range(self.start_level, self.end_level)])

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            fpn_conv = nn.Sequential(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.end_level - 1] if (i == 0 and self.extra_convs_on_inputs) 
                               else out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        # 使用SPIEM从第一个输入层提取位置信息
        spiem_output = self.spiem(inputs[0])

        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
            laterals[i - 1] += upsampled

        # 在融合过程中应用CIM模块
        outs = []
        for i in range(len(laterals)):
            lateral_output = laterals[i] + spiem_output  # 将SPIEM输出添加到lateral
            cim_output = self.cims[i](lateral_output)  # 应用CIM模块
            outs.append(self.fpn_convs[i](cim_output))

        if self.num_outs > len(outs):
            orig = inputs[self.start_level] if self.extra_convs_on_inputs else outs[-1]
            outs.append(self.fpn_convs[len(laterals)](orig))
            for i in range(len(laterals) + 1, self.num_outs):
                outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)