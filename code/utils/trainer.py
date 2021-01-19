
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time
import copy
from torchvision import models
import torchvision.transforms as transforms
from sklearn.metrics import make_scorer, accuracy_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            # Iterate over data.
            counter = 0
            for batch in dataloaders[phase]:
                labels,inputs = batch[0],batch[1]
                #print(labels)
                inputs = inputs.to(device)
                labels = labels.to(device) 
                #print(inputs)   
                # zero the parameter gradient
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    #print(outputs)
                    loss = criterion(outputs, labels)
                    # Get model predictions
                    _, preds = torch.max(outputs, 1)  
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.append(preds)
                all_labels.append(labels)

            epoch_loss = running_loss / dataloaders[phase].__len__()
            epoch_acc = running_corrects.double() / dataloaders[phase].__len__()
            all_labels = torch.cat(all_labels, 0)
            all_preds = torch.cat(all_preds, 0)
            epoch_weighted_acc = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), sample_weight=None) #compute_sample_weight(class_weights, all_labels.cpu().numpy())
            

            print('{} Loss: {:.4f} - Acc: {:.4f} - Weighted Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_weighted_acc))

            # deep copy the model
            if phase == 'val' and epoch_weighted_acc > best_acc:
                best_acc = epoch_weighted_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_weighted_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history