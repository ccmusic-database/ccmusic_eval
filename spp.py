from collections import OrderedDict
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class spp_layer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(spp_layer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x

class spp_net(nn.Module):
    """
    Expected input size is 64x64
    """

    def __init__(self, spp_level=3, num_classes=2):
        super(spp_net, self).__init__()
        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2**(i*2)
        #print(self.num_grids)

        self.conv_model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(3, 128, 3)), 
          ('relu1', nn.ReLU()),
          ('pool1', nn.MaxPool2d(2)),
          ('conv2', nn.Conv2d(128, 128, 3)),
          ('relu2', nn.ReLU()),
          ('pool2', nn.MaxPool2d(2)),
          ('conv3', nn.Conv2d(128, 128, 3)), 
          ('relu3', nn.ReLU()),
          ('pool3', nn.MaxPool2d(2)),
          ('conv4', nn.Conv2d(128, 128, 3)),
          ('relu4', nn.ReLU())
        ]))
        
        self.spp_layer = spp_layer(spp_level)

        self.linear_model = nn.Sequential(OrderedDict([
          ('fc1', nn.Linear(self.num_grids*128, 1024)),
          ('dropout', nn.Dropout(0.5)),
          ('fc1_relu', nn.ReLU()),
          ('dropout', nn.Dropout(0.5)),
          ('fc2', nn.Linear(1024, num_classes)),
        ]))

    def forward(self, x):
        x = self.conv_model(x)
        x = self.spp_layer(x)
        x = self.linear_model(x)
        return x