import numpy as np
import torch
from collections.abc import Callable
from typing import Optional, List

import utils

train_params = {'trim_or_pad_mode':'only_pad',
                'pad_mode':'zeros',
                'trim_mode':'fixed'}

def tac_algorithm(training_set, validation_set, base_image):
    """
    :param training_set:
    :param validation_set:
    :param base_image:

    """
    dim = len(training_set[0])
    nearest_odd_root = utils.nearest_odd_root(dim)
    op = trim_or_pad(training_set, dim, nearest_odd_root, mode=train_params['trim_or_pad_mode'])

    # training phase:
    # convert all the feature vectors into kernels -> convertion should be in torch.nn.Dataset
    # apply each kernel on the base image -> also this in the same pipeline
    # -> data pipeline with two of the above. (Dataset, DataLoader)
    # train a cnn with that

    # validation phase:
    # convert all the feature vectors into kernels
    # apply each kernel on the base image
    # evaluate
    
    pass

def feature_vector_to_kernel(features_vector: np.ndarray) -> np.ndarray:
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

    pass


def trim_or_pad(training_set, dim: int, nearest_odd: int, mode='default') -> str:
    """
    Decides if one should trim or pad the dataset's feature vector.

    :param training_set:
    :param dim:
    :param nearest_odd:
    :oaram mode: Which method we use to decide if trim or pad.
                 'default' - 
                 'only_trim' -
                 'only_pad' -
    
    :return: The desired operation between 'trim' or 'pad' from the utils.
    """
    if mode == 'only_trim':
        return 'trim'

    if mode == 'only_pad':
        return 'pad'
    
    if mode == 'default':
        raise NotImplementedError
    
    else:
        # TODO: raise error for using unsupported mode
        pass
