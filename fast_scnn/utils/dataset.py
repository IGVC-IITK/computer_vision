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
    def __init__(self, dataset_dir, transform=None):
        self.input_paths = []
        self.label_paths = []
        self.transform = transform
        # Assumes that the dataset_dir contains subdirectories with folders
        # named 'inputs' and 'labels', where each pair contains .jpg images in
        # the 'inputs' folder and corresponding .png images (with the same
        # name) in the 'labels' folder
        for root, dirs, files in os.walk(dataset_dir):
            if root[-6:] == 'inputs':
                for file in files:
                    if file[-4:] == '.jpg':
                        self.input_paths.append(os.path.join(root, file))
                        self.label_paths.append(os.path.join(root[:-6]+'labels', file[:-4]+'.png'))

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_img = Image.open(self.input_paths[idx]).convert('RGB')
        label_img = Image.open(self.label_paths[idx]).convert('P')
        seed = random.randrange(sys.maxsize)
        if self.transform['inputs']:
            random.seed(seed)
            input_img = self.transform['inputs'](input_img)
        if self.transform['labels']:
            random.seed(seed)
            label_img = self.transform['labels'](label_img)
            label_img = torch.round(label_img*255.0).long()

        return input_img, label_img
