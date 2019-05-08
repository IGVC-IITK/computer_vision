#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision import transforms, datasets, utils
from PIL import Image
import PIL

import os
import random
import sys


# In[2]:


class IGVCDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, labels_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.files_names = os.listdir(imgs_dir)
        
    def __len__(self):
        return len(self.files_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_dir, self.files_names[idx])
        label_path = os.path.join(self.labels_dir, self.files_names[idx])
        
        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('1')
        
        img = img.filter(PIL.ImageFilter.BLUR)
        seed = random.randrange(sys.maxsize)
        img = transforms.Grayscale(num_output_channels=1)(img)
        if self.transform:
            random.seed(seed)
            img = self.transform(img)
            random.seed(seed)
            label = self.transform(label)
        
        img = transforms.Normalize(mean=[0.5], std=[0.5])(img)
        return img, label

