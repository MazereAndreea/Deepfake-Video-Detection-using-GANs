import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import torch
import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64 * 64 * 1, 512),  # Adjusted for 1 channels
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.model(x)
    def lossFunction(self):
        criterion = nn.BCEWithLogitsLoss()
        return
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def builModel(self):

        return

    def trainModel(self,inputX,inputY):

        return
