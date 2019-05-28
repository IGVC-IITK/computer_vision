#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import os
from shutil import copy2


# In[2]:


data_folder = './data'
split_folders = ['./data_train', './data_val', './data_test']
ratios = [0.6, 0.3, 0.1]            # Train : Val : Test
assert ((0.9999 < sum(ratios)) and (sum(ratios) < 1.0001)), 'Sum of ratios not = 1.0'


# In[3]:


def safe_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def copy_img_files(file, src_dir, data_folder, split_folder):
    dst_dir = split_folder + src_dir[len(data_folder):]
    safe_mkdir(dst_dir)
    copy2(os.path.join(src_dir, file), os.path.join(dst_dir, file))


# In[5]:


random.seed(1337) # For reproducability of results


# In[6]:

input_dirs = []
input_files = []
label_dirs = []
label_files = []
# Assumes that the dataset_dir contains subdirectories with folders
# named 'inputs' and 'labels', where each pair contains .jpg images in
# the 'inputs' folder and corresponding .png images (with the same
# name) in the 'labels' folder
for root, dirs, files in os.walk(data_folder):
    if root[-6:] == 'inputs':
        for file in files:
            if file[-4:] == '.jpg':
                input_dirs.append(root)
                input_files.append(file)
                label_dirs.append(root[:-6]+'labels')
                label_files.append(file[:-4]+'.png')

for i in range(len(input_files)):
    rand = random.uniform(0, 1)
    if rand >= ratios[1] + ratios[2]:
        copy_img_files(input_files[i], input_dirs[i], data_folder, split_folders[0])
        copy_img_files(label_files[i], label_dirs[i], data_folder, split_folders[0])
    elif rand < ratios[1] + ratios[2] and rand > ratios[2]:
        copy_img_files(input_files[i], input_dirs[i], data_folder, split_folders[1])
        copy_img_files(label_files[i], label_dirs[i], data_folder, split_folders[1])
    else:
        copy_img_files(input_files[i], input_dirs[i], data_folder, split_folders[2])
        copy_img_files(label_files[i], label_dirs[i], data_folder, split_folders[2])
