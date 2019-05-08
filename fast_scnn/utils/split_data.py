#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import os
from shutil import copy2


# In[2]:


data_folder = '../data'
split_folders = ['../data_train', '../data_val', '../data_test']
sub_folders = ['images', 'labels']
ratio = [0.3, 0.2, 0.5]            # Train : Val : Test
assert sum(ratio) == 1.0, 'Sum of ratio not 1'


# In[3]:


def safe_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def copy_img_files(file, src_dir, dst_dir, sub_dirs):
    if file.endswith('.jpg'):
        for subd in  sub_dirs:
            copy2(os.path.join(src_dir, subd, file), os.path.join(dst_dir, subd))


# In[4]:


for splitd in split_folders:
    for subd in sub_folders:
        safe_mkdir(os.path.join(splitd, subd))


# In[5]:


random.seed(1337) # For reproducability of results


# In[6]:


files = os.listdir(os.path.join(data_folder, subd))
for f in files:
    rand = random.uniform(0, 1)
    if rand >= ratio[1] + ratio[2]:
        copy_img_files(f, data_folder, split_folders[0], sub_folders)
    elif rand < ratio[1] + ratio[2] and rand > ratio[2]:
        copy_img_files(f, data_folder, split_folders[1], sub_folders)
    else:
        copy_img_files(f, data_folder, split_folders[2], sub_folders)


# In[7]:


parent_count = len(os.listdir(os.path.join(data_folder, subd)))
tot_child_count = 0
print('No of files in parent dir :', parent_count)
for splitd in split_folders:
    count = len(os.listdir(os.path.join(splitd, subd)))
    tot_child_count = tot_child_count + count
    print('No of files in', splitd, ':', count)
assert(parent_count == tot_child_count), 'Total number of files in child dirs != No of files in parent dir.'
print('........................................................................')
print('Total number of files in child dirs = No of files in parent dir =', parent_count)

