from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, PPFConv
import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_mean_pool
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
class MVCNN(torch.nn.Module):
    def __init__(self, num_classes=40, pretrained=True):
        super(MVCNN, self).__init__()
        resnet = models.resnet34(pretrained = pretrained)
        fc_in_features = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_in_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, inputs): # inputs.shape = samples x views x height x width x channels
        inputs = inputs.transpose(0, 1)
        view_features = [] 
        for view_batch in inputs:
            view_batch = self.features(view_batch) #one view at a time
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            view_features.append(view_batch) #stack them
            
        pooled_views, _ = torch.max(torch.stack(view_features), 0) #maxpool
        outputs = self.classifier(pooled_views) #classifier
        return outputs
def get_model(classes = 40,pretrained = True):
    return MVCNN(classes,pretrained)