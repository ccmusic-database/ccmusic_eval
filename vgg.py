import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math

class VGG(nn.Module):
    def __init__(self, num_classes = 1000):
        super(VGG, self).__init__()

        # convolution layers
        self.features = nn.Sequential(
            # conv3-64_1 Input:[3,224,224] Output:[64,224,224]
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            # conv3-64_2 Input:[64,224,224] Output:[64,224,224]
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),

            # pool1 Output:[64,112,112]
            nn.MaxPool2d(kernel_size=2, stride=2),
                        
            # conv3-128_1 Input:[64,112,112] Output:[128,112,112]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            # conv3-128_2 Input:[128,112,112] Output:[128,112,112]
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),

            # pool2 Output:[128,56,56]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv3-256_1 Input:[128,56,56] Output:[256,56,56]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # conv3-256_2 Input:[256,56,56] Output:[256,56,56]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            # conv3-256_3 Input:[256,56,56] Output:[256,56,56]
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),

            # pool3 Output:[256,28,28]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv3-512_1 Input:[256,28,28] Output:[512,28,28]
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # conv3-512_2 Input:[512,28,28] Output:[512,28,28]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # conv3-512_3 Input:[512,28,28] Output:[512,28,28]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),

            # pool4 Output:[512,14,14]
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv3-512_4 Input:[512,14,14] Output:[512,14,14]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # conv3-512_5 Input:[512,14,14] Output:[512,14,14]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            # conv3-512_6 Input:[512,14,14] Output:[512,14,14]
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),

            # pool5 Output:[512,7,7]
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # fully connected layers
        self.classifier = nn.Sequential(
            # FC1
            nn.Dropout(p=0.5),  
            nn.Linear(32768, 2048), 
            nn.ReLU(True),
            # FC2
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            # FC3
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        # print(x.size())
        x = self.classifier(x)
        return x
