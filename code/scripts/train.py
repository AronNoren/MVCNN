import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import models
import torchvision.transforms as transforms


import pickle
from joblib import dump, load

import warnings
warnings.filterwarnings('ignore')

from utils.dataloader import get_data
from utils.trainer import train_model
criterion = nn.CrossEntropyLoss()
from models.MVCNN import get_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#get model
model = get_model(pretrain = False)

# FREEZE THE WEIGHTS IN THE FEATURE EXTRACTION BLOCK OF THE NETWORK (I.E. RESNET BASE)
for param in model.features.parameters():
    param.requires_grad = False
print("getting data")
data_loaders = get_data()
print("getting model")
model.to(device)
EPOCHS = 5
#weight = torch.tensor([0.327, 0.586, 2.0, 2.572, 26.452, 15.0]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

print("TRAINING CLASSIFIER")

model, val_acc_history = train_model(model=model, dataloaders=data_loaders, criterion=criterion, optimizer=optimizer, num_epochs=EPOCHS)

torch.save(model.state_dict(), 'code/models/saved_models/mvcnn_stage_1.pkl')

for param in model.parameters():
    param.requires_grad = True

print("FINE TUNING")

EPOCHS = 3
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005) # We use a smaller learning rate

model, val_acc_history = train_model(model=model, dataloaders=data_loaders, criterion=criterion, optimizer=optimizer, num_epochs=EPOCHS)

torch.save(model.state_dict(), 'code/models/saved_models/mvcnn_stage_fine.pkl')