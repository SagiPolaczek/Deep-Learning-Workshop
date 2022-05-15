import math
import numpy as np


def nearest_odd_root(d: int) -> int:
    """
    Computes the nearest odd s

    TODO: make the code more clean, return sqrt_d + ((sqrt + 1) % 2) or something similar

    :return: The nearest odd root from above. for example:
             For d = 25 it will return 5.
             For d = 26 it will return 7.
    """
    sqrt_d = math.sqrt(d)

    if sqrt_d.is_integer():
        if sqrt_d % 2 == 1:
            # sqrt_d is odd
            return int(sqrt_d)
        else:
            # sqrt_d is even
            return int(sqrt_d + 1)
    else:
        # sqrt_d is not an integer
        closet_int_from_above = int(sqrt_d) + 1
        if closet_int_from_above % 2 == 1:
            # is odd
            return closet_int_from_above
        else:
            # is even
            return closet_int_from_above + 1


def pad(feature_vector: np.ndarray, k: int, method='random'):
    """
    Pads the feature vector
    :param feature_vector:
    :param k:
    :param method: Method using for the padding:
                   'random' - 
                   'zeros' - 
    """

    pass

    

def trim(feature_vector: np.ndarray, k: int, method='random'):
    """
    Trim the feature_vector so 
    """
    pass
