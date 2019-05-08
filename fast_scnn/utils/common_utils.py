#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
from torchvision import transforms, datasets, utils
from PIL import Image


# In[6]:


def output_mask(img, model, device):
    img = img.to(device)
    output = model(img)
    mask = torch.argmax(output, 1).detach().cpu()
    return mask


# In[7]:


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print('Model saved to ', path)


# In[8]:


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print('Loaded Model from ', path)


# In[9]:


def img_to_mask(img_path, model, transforms, device):
    img = Image.open(img_path).convert('RGB')
    img = transforms(img)
    img = img.unsqueeze(0).to(device)
    output = model(img)
    mask = torch.argmax(output, 1).detach().cpu()
    return mask


# In[ ]:




