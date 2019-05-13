#!/usr/bin/env python
# coding: utf-8

# In[2]:


import copy
import time
import torch
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


# In[3]:


def train_net(model, dataloaders, dataset_sizes, criterion, optimizer, 
                scheduler, device, num_epochs=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('Inf')
    epoch_numbers = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).long()[:, -1, :, :]
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(input=outputs, target=labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase] / (inputs.size(2)*inputs.size(3))
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
            if phase == 'val':
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)
        epoch_numbers.append(epoch)
        clear_output(wait=True)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print('train: loss: {:.4f} acc: {:.2f}%'.format(
            train_losses[epoch], train_accuracies[epoch]*100.0))
        print('val:   loss: {:.4f} acc: {:.2f}%'.format(
            val_losses[epoch], val_accuracies[epoch]*100.0))
        if epoch > 0:
            plt.plot(epoch_numbers, train_losses, 'r-', epoch_numbers, val_losses, 'g-')
            plt.axis([0, num_epochs, 0, 1])
            plt.show()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed//60, time_elapsed%60))
    print('Best Val Loss: {:2f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    print('Done Training.')
    return model

