import numpy as np
import torch
import torch.nn as nn
from torchvision import models
#import lightning.pytorch as pl
import pytorch_lightning as pl

class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.change_head_net()

        self.loss_fn = torch.nn.L1Loss()
        
        # pytorch imagenet calculated mean/std
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

    def forward(self, image):
        return self.model(image)

    def model_parameters(self):
        return self.model.parameters()

    def change_head_net(self):
        num_ftrs = self.model.classifier[-1].in_features

        head_net = nn.Sequential(
            nn.Linear(num_ftrs, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 51, bias=True),
        )

        self.model.classifier[-1] = head_net
    
    def compute_metrics(pred_trajectory, target_trajectory):
        # L1 and L2 distance: matrix of size BSx40x3
        L1_loss = torch.abs(pred_trajectory - target_trajectory)
        L2_loss = torch.pow(pred_trajectory - target_trajectory, 2)

        # BSx40x3 -> BSx3 average over the predicted points
        L1_loss = L1_loss.mean(axis=1)
        L2_loss = L2_loss.mean(axis=1)

        # split into losses for each axis and an avg loss across 3 axes
        # All returned tensors have shape (BS)
        return {
                'L1_loss':   L1_loss.mean(axis=1),
                'L1_loss_x': L1_loss[:, 0],
                'L1_loss_y': L1_loss[:, 1],
                'L1_loss_z': L1_loss[:, 2],
                'L2_loss':   L2_loss.mean(axis=1),
                'L2_loss_x': L2_loss[:, 0],
                'L2_loss_y': L2_loss[:, 1],
                'L2_loss_z': L2_loss[:, 2]}
