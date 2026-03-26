import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class SEFPN(BaseModule):
    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1, 
                 add_extra_convs=False, extra_convs_on_inputs=True, norm_cfg=None, act_cfg=None):
        super(SEFPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.end_level = self.num_ins if end_level == -1 else end_level
        self.backbone_end_level = self.end_level
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
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

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1] if (i == 0 and self.extra_convs_on_inputs) 
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
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode='nearest')
            laterals[i - 1] += upsampled

        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        if self.num_outs > len(outs):
            orig = inputs[self.start_level] if self.extra_convs_on_inputs else outs[-1]
            outs.append(self.fpn_convs[len(laterals)](orig))
            for i in range(len(laterals) + 1, self.num_outs):
                outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)
