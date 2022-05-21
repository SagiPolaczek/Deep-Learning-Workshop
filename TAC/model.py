import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


class TACDataset(Dataset):
    """
    
    """

    def __init__(self, file):
        """
        
        :param file:
        """
        self.file = file

        # TODO:
        # 1) Read file
        # 2) Save len in self._len

        self._len = 0
        pass

    def __len__(self) -> int:
        """
        returns dataset length  
        """
        return self._len

    def __getitem__(self, index):
        """
        Returns the 'index' item in the dataset * after applying the preprocessing pipeline *

        :param index:
        """

        # read the 'index' feature vector
        # transform it to kernel
        # apply it on the base image
        # return the convolved image
        pass