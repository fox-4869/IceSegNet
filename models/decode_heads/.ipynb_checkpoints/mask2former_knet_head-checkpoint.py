import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention, build_transformer_layer
from mmengine.logging import print_log

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.utils import SampleList


@MODELS.register_module()
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dims=256, num_layers=6, num_heads=8, feedforward_channels=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dims,
                nhead=num_heads,
                dim_feedforward=feedforward_channels,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@MODELS.register_module()
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dims=256, num_layers=6, num_heads=8, feedforward_channels=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dims,
                nhead=num_heads,
                dim_feedforward=feedforward_channels,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt


@MODELS.register_module()
class DynamicMaskLearning(nn.Module):
    def __init__(self, in_channels, out_channels, num_proposals, num_heads):
        super().__init__()
        self.dynamic_mask = nn.Linear(in_channels, num_proposals)
        self.self_attention = MultiheadAttention(in_channels, num_heads)

    def forward(self, x, mask_proposals):
        dynamic_masks = self.dynamic_mask(x)
        dynamic_masks = dynamic_masks.sigmoid()
        attended_masks = self.self_attention(mask_proposals, dynamic_masks, dynamic_masks)
        return attended_masks


@MODELS.register_module()
class MultiScaleFeatureIntegration(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.convs.append(
                ConvModule(in_channels, out_channels, kernel_size=3, padding=1)
            )
        self.fusion_conv = ConvModule(len(in_channels_list) * out_channels, out_channels, kernel_size=1)

    def forward(self, features):
        multi_scale_features = [conv(feat) for conv, feat in zip(self.convs, features)]
        fusion_features = torch.cat(multi_scale_features, dim=1)
        return self.fusion_conv(fusion_features)


@MODELS.register_module()
class MaskDynamicUpdator(nn.Module):
    def __init__(self, in_channels, feat_channels, out_channels):
        super().__init__()
        self.dynamic_layer = nn.Linear(in_channels, feat_channels * 2)
        self.norm_in = nn.LayerNorm(feat_channels)
        self.fc_layer = nn.Linear(feat_channels, out_channels)

    def forward(self, update_feature, input_feature):
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[:, :self.fc_layer.in_features]
        param_out = parameters[:, self.fc_layer.in_features:]
        input_in = param_in * input_feature
        input_out = param_out * input_feature
        fused_features = self.fc_layer(self.norm_in(input_in + input_out))
        return fused_features


@MODELS.register_module()
class KNetImproved(BaseDecodeHead):
    def __init__(self, 
                 in_channels_list=[256, 512, 1024, 2048],
                 embed_dims=256, 
                 num_classes=150,
                 num_proposals=100, 
                 num_layers=6, 
                 num_heads=8, 
                 feedforward_channels=1024, 
                 dropout=0.1,
                 **kwargs):
        super(KNetImproved, self).__init__(**kwargs)

        # Transformer Encoder-Decoder 机制
        self.encoder = TransformerEncoder(embed_dims, num_layers, num_heads, feedforward_channels, dropout)
        self.decoder = TransformerDecoder(embed_dims, num_layers, num_heads, feedforward_channels, dropout)

        # 动态 mask 学习
        self.dynamic_mask = DynamicMaskLearning(embed_dims, num_classes, num_proposals, num_heads)

        # 多尺度特征整合
        self.multi_scale_fusion = MultiScaleFeatureIntegration(in_channels_list, embed_dims)

        # mask 动态更新机制
        self.mask_updator = MaskDynamicUpdator(embed_dims, feat_channels=64, out_channels=embed_dims)

    def forward(self, inputs):
        # 多尺度特征整合
        multi_scale_features = self.multi_scale_fusion(inputs)

        # Transformer Encoder 对特征进行编码
        encoded_features = self.encoder(multi_scale_features)

        # 动态 mask 生成和更新
        dynamic_mask_proposals = torch.zeros(encoded_features.size(0), 100, encoded_features.size(-1), device=encoded_features.device)  # 初始化mask proposals
        mask_preds = []
        for _ in range(6):  # 迭代更新mask
            dynamic_masks = self.dynamic_mask(encoded_features, dynamic_mask_proposals)
            dynamic_mask_proposals = self.mask_updator(dynamic_masks, dynamic_mask_proposals)
            mask_preds.append(dynamic_mask_proposals)

        # Transformer Decoder 逐步解码
        decoded_features = self.decoder(encoded_features, dynamic_mask_proposals)

        # 语义分割预测
        seg_logits = F.interpolate(self.cls_seg(decoded_features), size=inputs[0].shape[2:], mode='bilinear', align_corners=False)

        return seg_logits