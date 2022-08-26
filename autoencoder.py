import math
from turtle import forward
from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super(SingleConv, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.single_conv(x)
        return x




class Encoder(nn.Module):
    """
    TODO
    """

    def __init__(self, in_channels: int, out_channels: int, verbose: bool = True):
        """
        """
        super().__init__()

        self._verbose = verbose
        
        self.layer1 = SingleConv(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.layer2 = SingleConv(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.layer3 = SingleConv(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.layer4 = SingleConv(in_channels=16, out_channels=out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Decoder(nn.Module):
    """
    TODO
    """

    def __init__(self, in_channels: int, out_channels: int, decode: bool = True, verbose: bool = True):
        """
        """
        super().__init__()
        self._verbose = verbose
        self._decode = decode

        self.layer1 = SingleConv(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.layer2 = SingleConv(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.layer3 = SingleConv(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.layer4 = SingleConv(in_channels=16, out_channels=out_channels, kernel_size=3, padding=1)



    def forward(self, x):

        if self._decode:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x


