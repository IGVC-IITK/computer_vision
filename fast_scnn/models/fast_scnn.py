#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn import functional as F


# #### Level 1 (Building Blocks)

# In[2]:


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, relu=True):
        super().__init__()
        if relu:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
                nn.BatchNorm2d(out_channels),   # Batch-norm => 0 bias convolution in previous layer
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False),
                nn.BatchNorm2d(out_channels)    # Batch-norm => 0 bias convolution in previous layer
            )

    def forward(self, x):
        return self.conv_block(x)               # Note that batch-norm requires x.size()[2:4] != (1, 1)


# In[3]:


class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1):
        super().__init__()
        self.ds_conv_block = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, relu=True),
            ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, relu=True)
        )
        
    def forward(self, x):
        return self.ds_conv_block(x)


# In[4]:


class InvertedResidualBlock(nn.Module):
    # As described in MobileNetV2
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, expand_ratio=6):
        super().__init__()
        inter_channels = round(in_channels*expand_ratio)
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        self.inv_res = nn.Sequential(
            ConvBlock(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, relu=True),
            ConvBlock(inter_channels, inter_channels, kernel_size, stride, padding, dilation, groups=inter_channels, relu=True),
            ConvBlock(inter_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, relu=False)
        )
        
    def forward(self, x):
        if self.use_res_connect:
            return (x + self.inv_res(x))
        else:
            return self.inv_res(x)
        


# In[5]:


class PyramidPoolingBlock(nn.Module):
    # As described in PSPNet (Pyramid Scene Parsing Network)
    def __init__(self, in_channels, out_channels, bin_sizes=[1, 2, 3, 6]):
        super().__init__()
        inter_channels = in_channels//len(bin_sizes)
        total_channels = in_channels + inter_channels*len(bin_sizes)
        self.levels = nn.ModuleList()
        for i in range(0, len(bin_sizes)):
            self.levels.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_sizes[i]),
                nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
            ))
        self.out_conv = ConvBlock(total_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, relu=True)

    def forward(self, x):
        spatial_dim = x.size()[2:4]
        feature_map = x
        for i in range(0, len(self.levels)):
            level = self.levels[i](feature_map)
            level = F.interpolate(level, spatial_dim, mode='bilinear', align_corners=True)
            x = torch.cat([x, level], dim=1)
        return self.out_conv(x)


# ### Level 2 (Modules)

# In[6]:


class LearningToDownsample(nn.Module):
    def __init__(self, in_channels, alpha):
        super().__init__()
        self.learning_to_downsample = nn.Sequential(
            ConvBlock(in_channels, round(32*alpha), kernel_size=3, stride=2, padding=1, dilation=1, groups=1, relu=True),
            DSConvBlock(round(32*alpha), round(48*alpha), kernel_size=3, stride=2, padding=1, dilation=1),
            DSConvBlock(round(48*alpha), round(64*alpha), kernel_size=3, stride=2, padding=1, dilation=1)
        )
        
    def forward(self, x):
        return self.learning_to_downsample(x)


# In[7]:


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.global_feature_extractor = nn.Sequential(
            # Sequence 1
            InvertedResidualBlock(round(64*alpha), round(64*alpha), kernel_size=3, stride=2, padding=1, dilation=1, expand_ratio=6),
            InvertedResidualBlock(round(64*alpha), round(64*alpha), kernel_size=3, stride=1, padding=1, dilation=1, expand_ratio=6),
            InvertedResidualBlock(round(64*alpha), round(64*alpha), kernel_size=3, stride=1, padding=1, dilation=1, expand_ratio=6),
            # Sequence 2
            InvertedResidualBlock(round(64*alpha), round(96*alpha), kernel_size=3, stride=2, padding=1, dilation=1, expand_ratio=6),
            InvertedResidualBlock(round(96*alpha), round(96*alpha), kernel_size=3, stride=1, padding=1, dilation=1, expand_ratio=6),
            InvertedResidualBlock(round(96*alpha), round(96*alpha), kernel_size=3, stride=1, padding=1, dilation=1, expand_ratio=6),
            # Sequence 3
            InvertedResidualBlock(round(96*alpha), round(128*alpha), kernel_size=3, stride=1, padding=1, dilation=1, expand_ratio=6),
            InvertedResidualBlock(round(128*alpha), round(128*alpha), kernel_size=3, stride=1, padding=1, dilation=1, expand_ratio=6),
            InvertedResidualBlock(round(128*alpha), round(128*alpha), kernel_size=3, stride=1, padding=1, dilation=1, expand_ratio=6),
            # Pyramid Pooling
            PyramidPoolingBlock(round(128*alpha), round(128*alpha), bin_sizes=[1, 2, 3, 6])
        )
        
    def forward(self, x):
        return self.global_feature_extractor(x)


# In[8]:


class FeatureFusion(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.conv_high_res_pw = nn.Conv2d(round(64*alpha), round(128*alpha), kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.conv_low_res_dw = ConvBlock(round(128*alpha), round(128*alpha), kernel_size=3, stride=1, padding=4, dilation=4, groups=round(128*alpha), relu=True)
        self.conv_low_res_pw = nn.Conv2d(round(128*alpha), round(128*alpha), kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, high_res_input, low_res_input):
        # First Branch
        high_res_input = self.conv_high_res_pw(high_res_input)
        # Second Branch
        spatial_dim = high_res_input.size()[2:4]
        low_res_input = F.interpolate(low_res_input, spatial_dim, mode='bilinear', align_corners=True)
        low_res_input = self.conv_low_res_dw(low_res_input)
        low_res_input = self.conv_low_res_pw(low_res_input)
        # Fusing Both
        return self.relu(high_res_input + low_res_input)


# In[9]:


class Classifier(nn.Module):
    def __init__(self, alpha, num_classes, dropout_prob):
        super().__init__()
        self.classifier = nn.Sequential(
            DSConvBlock(round(128*alpha), round(128*alpha), kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Dropout2d(dropout_prob),
            DSConvBlock(round(128*alpha), round(128*alpha), kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(round(128*alpha), num_classes, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm2d(num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)


# ## Level 3 (Whole model)

# In[10]:


class FastSCNN(nn.Module):
    def __init__(self, in_channels, width_multiplier, num_classes, dropout_prob=0.5):
        super().__init__()
        # alpha is the width_multiplier as described in the MobileNets paper
        alpha = width_multiplier
        self.learning_to_downsample = LearningToDownsample(in_channels, alpha)
        self.global_feature_extractor = GlobalFeatureExtractor(alpha)
        self.feature_fusion = FeatureFusion(alpha)
        self.classifier = Classifier(alpha, num_classes, dropout_prob)
        
    def forward(self, x):
        spatial_dim = x.size()[2:4]
        x = self.learning_to_downsample(x)
        y = self.global_feature_extractor(x)
        x = self.feature_fusion(x, y)
        x = self.classifier(x)
        return x
