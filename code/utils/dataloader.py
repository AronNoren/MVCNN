import sys
import os
parent_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_path)
from torch.utils.data import Dataset, DataLoader
from utils.dataset import MultiviewImgDataset
from torch.utils.data.sampler import SubsetRandomSampler
def get_data(split = 0.8):
    train_dataset = MultiviewImgDataset(root_dir = 'data/modelnet40/*/train')
    test_dataset = MultiviewImgDataset(root_dir = 'data/modelnet40/*/test')
    #print(train_dataset.__getitem__(0))
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    data_loaders = {'train': train_loader, 'val': val_loader}
    return data_loaders