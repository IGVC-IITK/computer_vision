#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn import functional as F


# #### Level 1

# In[2]:


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if x.size()[2:] != (1, 1):
            x = self.bn(x)
        x = self.activation(x)
        return x


# In[3]:


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, dilation=1):
        super().__init__()
        self.ds_conv = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels),
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        )
        
    def forward(self, x):
        return self.ds_conv(x)


# In[4]:


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        
        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.inv_res = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0, dilation=1, groups=1),
            ConvBlock(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=1, dilation=1, groups=hidden_dim),
            nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.inv_res(x)
        else:
            return self.inv_res(x)
        


# In[5]:


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = in_channels // 4
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.conv2= ConvBlock(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.conv3 = ConvBlock(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.conv4 = ConvBlock(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.out_conv = ConvBlock(in_channels=2*in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        
    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        fcn_feat_spatial_dim = x.size()[2:]
        
        pool1 = self.upsample(self.conv1(self.pool(x, 1)), fcn_feat_spatial_dim)
        pool2 = self.upsample(self.conv1(self.pool(x, 2)), fcn_feat_spatial_dim)
        pool3 = self.upsample(self.conv1(self.pool(x, 3)), fcn_feat_spatial_dim)
        pool4 = self.upsample(self.conv1(self.pool(x, 6)), fcn_feat_spatial_dim)
        
        x = torch.cat([x, pool1, pool2, pool3, pool4], dim=1)
        
        x = self.out_conv(x)
        return x


# ### Level 2

# In[6]:


class LearnDownsampling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.learn_downsampling = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=32, stride=2),
            DSConv(in_channels=32, out_channels=48, kernel_size=3, stride=2, padding=1, dilation=1),
            DSConv(in_channels=48, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1)
        )
        
    def forward(self, x):
        return self.learn_downsampling(x)


# In[7]:


class GlobalFeatExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_res_block1 = nn.Sequential(
            InvertedResidual(in_channels=64, out_channels=64, stride=2, expand_ratio=6),
            InvertedResidual(in_channels=64, out_channels=64, stride=1, expand_ratio=6),
            InvertedResidual(in_channels=64, out_channels=64, stride=1, expand_ratio=6)
        )
        self.inv_res_block2 = nn.Sequential(
            InvertedResidual(in_channels=64, out_channels=96, stride=2, expand_ratio=6),
            InvertedResidual(in_channels=96, out_channels=96, stride=1, expand_ratio=6),
            InvertedResidual(in_channels=96, out_channels=96, stride=1, expand_ratio=6)
        )
        self.inv_res_block3 = nn.Sequential(
            InvertedResidual(in_channels=96, out_channels=128, stride=1, expand_ratio=6),
            InvertedResidual(in_channels=128, out_channels=128, stride=1, expand_ratio=6),
            InvertedResidual(in_channels=128, out_channels=128, stride=1, expand_ratio=6)
        )
        self.pyramid_pool = PyramidPooling(128, 128)
        
    def forward(self, x):
        x = self.inv_res_block3(self.inv_res_block2(self.inv_res_block1(x)))
        return self.pyramid_pool(x)


# In[8]:


class FeatFusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=128)
        self.conv_low_res = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_high_res = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, high_res_input, low_res_input):
        final_size = high_res_input.size()[2:]
        low_res_input = F.interpolate(low_res_input, size=final_size, mode='bilinear', align_corners=True)
        low_res_input = self.conv1(low_res_input)
        low_res_input = self.conv_low_res(low_res_input)
        
        high_res_input = self.conv_high_res(high_res_input)
        fused = torch.add(high_res_input, low_res_input)
        return self.relu(fused)


# In[10]:


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier_conv = nn.Sequential(
            DSConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            DSConv(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        )
        
    def forward(self, x):
        return self.classifier_conv(x)


# ## Level 3 (Whole model)

# In[11]:


class FastSCNN(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.learning_to_ds = LearnDownsampling(in_channel)
        self.global_feat_ext = GlobalFeatExtractor()
        self.feat_fuse = FeatFusionModule()
        self.classifier = Classifier(num_classes)
        
    def forward(self, x):
        in_size = x.size()[2:]
        shared = self.learning_to_ds(x)
        x = self.global_feat_ext(shared)
        x = self.feat_fuse(shared, x)
        x = self.classifier(x)
        x = F.interpolate(x, size=in_size, mode='bilinear', align_corners=True)
        return x


# In[ ]:




