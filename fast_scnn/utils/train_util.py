#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import time
import torch
from torch.nn import functional as F
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


# In[2]:


def train_net(model, dataloaders, datasizes, classes,
                criterion, optimizer, scheduler, device, num_epochs=20):
    since = time.time()

    losses = {'train': [], 'val': []}
    ious = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('Inf')
    best_epoch = -1
    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:

            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_intersection = [0]*len(classes)
            running_union = [0]*len(classes)

            for inputs, labels in dataloaders[phase]:
                # Forward and backward passes
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    spatial_dim = outputs.size()[2:4]
                    labels = F.interpolate(labels.float(), spatial_dim, mode='nearest')[:, 0, :, :].long()
                    preds = torch.argmax(outputs, 1)
                    loss = criterion(input=outputs, target=labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # For calculating losses and IoUs
                running_loss += loss.item()*inputs.size(0)
                for i in range(len(classes)):
                    running_intersection[i] += torch.sum((preds == i) & (labels == i))
                    running_union[i] += torch.sum((preds == i) | (labels == i))
            epoch_loss = running_loss / datasizes[phase]
            epoch_iou = []
            for i in range(len(classes)):
                epoch_iou.append(running_intersection[i].float()/running_union[i].float())
            losses[phase].append(epoch_loss)
            ious[phase].append(epoch_iou)

            if phase == 'val' and epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        # Reporting losses and IoUs
        clear_output(wait=True)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            print('{:s}:'.format(phase))
            print('  loss: {:.4f}'.format(losses[phase][epoch]))
            print('  IoUs:')
            for i in range(len(classes)):
                print('    {:15s}: {:.2f}'.format(classes[i], ious[phase][epoch][i]*100.0))
        if epoch > 0:
            plt.plot(range(epoch + 1), losses['train'], 'r-', 
                        range(epoch + 1), losses['val'], 'g-')
            plt.axis([0, num_epochs - 1, 0, 2]); plt.show()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed//60, time_elapsed%60))
    print('Best val loss: {:.4f} (epoch {:d})'.format(best_val_loss, best_epoch))
    print('IoUs for best val loss:')
    for i in range(len(classes)):
        print('  {:15s}: {:.2f}'.format(classes[i], ious['val'][best_epoch][i]*100.0))

    best_model = model     
    best_model.load_state_dict(best_model_wts)

    return best_model, model
