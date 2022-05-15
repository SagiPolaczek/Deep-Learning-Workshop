import numpy as np
import torch
from collections.abc import Callable

import utils



def tac_algorithm(training_set, validation_set, base_image):
    """

    """
    dim = len(training_set[0])
    nearest_odd = utils.nearest_odd_root(dim)
    op = trim_or_pad(training_set, dim, nearest_odd, mode='only_pad')

    
    pass

def feature_vector_to_kernel(feature_vector: np.ndarray, nearest_odd: int, op: Callable[[], np.ndarray]) -> np.ndarray:
    """
    Converts a feature vector into a kerenl with a size (k x k) when k := nearest_odd using the supplied op.

    :param feature_vector:
    :param nearest_odd:
    :param od:
    :return: The resulted kernel.
    """
    
    mean_value = np.mean(feature_vector)
    fixed_vector = op(feature_vector, nearest_odd, method='zeros')



    


    pass


def trim_or_pad(training_set, dim: int, nearest_odd: int, mode='default') -> Callable[[], numpy.ndarray]:
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
        return utils.trim

    if mode == 'only_pad':
        return utils.pad
    
    if mode == 'default':
        raise NotImplementedError
    
    else:
        # TODO: raise error for using unsupported mode
        pass

def get_vector_feature_dimension(training_set):
    """
    :param training_set:
    :return:
    """
    pass

