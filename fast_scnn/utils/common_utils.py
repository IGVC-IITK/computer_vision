#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision import transforms, datasets, utils
from PIL import Image


# In[7]:


def save_params(model, path):
    torch.save(model.state_dict(), path)
    print('Model parameters saved to ', path)


# In[8]:


def load_params(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print('Loaded model parameters from ', path)

