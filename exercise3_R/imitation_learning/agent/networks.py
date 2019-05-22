import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3): 
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network

    def forward(self, x):
        # TODO: compute forward pass
        return x

