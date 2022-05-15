import numpy
import torch
from collections.abc import Callable

import utils



def tac_algorithm(training_set, validation_set, base_image):
    """

    """
    

    pass

def feature_vector_to_kernel(feature_vector: numpy.ndarray):
    """

    """
    dim = len(feature_vector)
    mean_value = numpy.mean(feature_vector)
    nearest_odd = utils.nearest_odd_root(dim)
    op = trim_or_pad()
    


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
    

def get_feature_dimension(training_set):
    """
    """
    pass

