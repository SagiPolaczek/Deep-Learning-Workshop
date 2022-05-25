import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import skimage
import scipy.signal as signal
from typing import Optional

import tac
import utils


class SchoolDataset(Dataset):
    """
    
    """

    def __init__(self, file_path: str, base_image_path: Optional[str] = None, train: bool = True):
        """
        TODO added train/valid partition

        """
        self._file_path = file_path
        self._base_image_path = base_image_path

        self._len = pd.read_csv(file_path).shape[0]
        self._op = 'trim'
        self._features_dim = 29

        # Define base image
        # Can be read from path or supplied externaly
        self._base_image = skimage.data.coins()

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
        # Read the 'index' feature vector
        df = pd.read_csv(self._file_path)
        item = df.iloc[index].values

        features_vector = item[:-1]
        label = item[-1]

        # transform it to kernel
        features_vector = np.delete(features_vector, [8,9,10,13])
        kernel = tac.feature_vector_to_kernel(features_vector=features_vector, k=5, mode='default')

        # apply it on the base image
        image = signal.convolve2d(self._base_image, kernel)
        image = utils.normalize(image)

        # return the convolved image
        sample = {'image': image, 'label': label}
        
        return sample
