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
from sklearn.model_selection import train_test_split
from torchvision import transforms


class HiggsDataset(Dataset):
    """
    
    """

    def __init__(self, file_path: str, base_image_path: Optional[str] = None, train: bool = True, transform: Optional[transforms.Compose] = None):
        """


        """
        self._file_path = file_path
        self._base_image_path = base_image_path
        self._transform = transform

        self._df = pd.read_csv(file_path)
        self._op = 'trim'
        self._features_dim = 30

        if base_image_path is None:
            # Define base image
            # Can be read from path or supplied externaly
            # self._base_image = skimage.data.coins() # 90%
            # self._base_image = skimage.data.cell() # 90%
            self._base_image = skimage.data.shepp_logan_phantom() # 94%

        # Split into training data and validation data
        samples_ids = self._df.index.tolist()
        train_samples, val_samples = train_test_split(samples_ids, test_size=0.25, random_state=42)

        # Define the dataframe and the dataset's length
        if train:
            self._df = self._df.iloc[train_samples]

        if not train:
            self._df = self._df.iloc[val_samples]

        self._len = self._df.shape[0]
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
        item = self._df.iloc[index].values

        features_vector = item[1:-2]
        label = {'s':0, 'b':1}[item[-1]]
        weight = item[-2]

        # print(f"feature vector shape {features_vector.shape}")

        # transform it to kernel
        features_vector = np.delete(features_vector, [7,11,13,17,23])  # TODO
        kernel = feature_vector_to_kernel(features_vector=features_vector, k=5, mode='default')

        # apply it on the base image
        # print(self._base_image.dtype)
        # print(type(self._base_image))
        # print(kernel.dtype)
        # print(type(kernel))
        # print(f"base image shape {self._base_image.shape}")
        # print(f"kernel shape {kernel.shape}")
        image = signal.convolve2d(self._base_image, kernel)
        image = normalize(image)

        # Exapnd to 3 channels to fit ResNet
        image = np.expand_dims(image, axis=-1)
        image = image.repeat(3, axis=-1)

        # apply transform
        if self._transform:
            image = self._transform(image)
            image = image.float()

        # return the convolved image
        sample = {'image': image, 'label': label, 'weight': weight}
        
        return sample



########

def feature_vector_to_kernel(features_vector: np.ndarray, k: int, mode: str = 'default') -> np.ndarray:
    """
    Converts a feature vector into a kerenl with a size (k x k) when k := nearest_odd using the supplied op.

    :param feature_vector:
    :param nearest_odd:
    :param od:
    :return: The resulted kernel.
    """

    # assert op in ['trim', 'pad'], f"op should be 'trim' or 'pad'. got: {op}"

    # mean_value = np.mean(features_vector)

    # if op == 'pad':
    #     feature_vector = utils.pad(features_vector, nearest_odd, train_params['pad_mode'])
    # if op == 'trim':
    #     feature_vector = utils.trim(features_vector, nearest_odd, train_params['trim_mode'])
    
    # now the vector with the right size (k^2)
    # should we convert it to a (k x k) kernel? (img/tensor etc. not a vector)

    if mode == 'default':
        mean_value = np.mean(features_vector)

        features_vector = features_vector - mean_value

    kernel = features_vector.reshape((k, k))
    kernel = kernel.astype(float)
    return kernel


def normalize(image: np.ndarray) -> np.ndarray:
    """
    
    """
    min_value = image.min()
    image = image - min_value
    image = image / image.max()

    return image
