import torch
import torch.nn as nn

from mmseg.registry import MODELS
from mmseg.models.decode_heads import FCNHead
from mmseg.models.necks import FPN

import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS
from ..utils import UpConvBlock, Upsample


@MODELS.register_module()
class UNetplus(BaseModule):
    def __init__(self,
                 num_classes=3,
                 in_channels=3,
                 init_channels=32,
                 dropout=0.5,
                 **kwargs):
        super(UNetplus, self).__init__(**kwargs)
        
        # Encoder blocks
        self.conv1 = self._conv_block(in_channels, init_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = self._conv_block(init_channels, init_channels*2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = self._conv_block(init_channels*2, init_channels*4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = self._conv_block(init_channels*4, init_channels*8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(init_channels*8, init_channels*16)
        
        # Decoder blocks
        self.upconv4 = nn.ConvTranspose2d(init_channels*16, init_channels*8, 2, stride=2)
        self.conv4d = self._conv_block(init_channels*16, init_channels*8)
        
        self.upconv3 = nn.ConvTranspose2d(init_channels*8, init_channels*4, 2, stride=2)
        self.conv3d = self._conv_block(init_channels*8, init_channels*4)
        
        self.upconv2 = nn.ConvTranspose2d(init_channels*4, init_channels*2, 2, stride=2)
        self.conv2d = self._conv_block(init_channels*4, init_channels*2)
        
        self.upconv1 = nn.ConvTranspose2d(init_channels*2, init_channels, 2, stride=2)
        self.conv1d = self._conv_block(init_channels*2, init_channels)
        
        # Final output
        self.final_conv = nn.Conv2d(init_channels, num_classes, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.conv1(x)
        enc2 = self.conv2(self.pool1(enc1))
        enc3 = self.conv3(self.pool2(enc2))
        enc4 = self.conv4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder path
        dec4 = self.conv4d(torch.cat([self.upconv4(bottleneck), enc4], dim=1))
        dec3 = self.conv3d(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.conv2d(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.conv1d(torch.cat([self.upconv1(dec2), enc1], dim=1))
        
        return self.final_conv(dec1)