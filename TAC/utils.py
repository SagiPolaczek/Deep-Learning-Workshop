import math
import numpy as np



def nearest_odd_root(d: int) -> int:
    """
    Computes the nearest odd root to 'd' from above (rounded up).

    TODO: make the code more clean, return sqrt_d + ((sqrt + 1) % 2) or something similar
    :param d:
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


def pad(feature_vector: np.ndarray, k: int, mode='zeros') -> None:
    """
    Pads the feature vector so that after padding it's dimension will be k^2.
    :param feature_vector:
    :param k: 
    :param method: Method using for the padding:
                   'random' - 
                   'zeros' - 
    
    :return: The feature vector after it has been padded to length k^2
    """
    if mode != 'zeros':
        raise NotImplementedError
    
    # Get how much should pad
    pad_amount = int(math.pow(k, 2) - len(feature_vector))

    assert pad_amount >= 0, "k^2 < dim !"

    
    if pad_amount % 2 == 0:
        # pad equally on both sides
        return np.pad(feature_vector, (int(pad_amount/2),), 'constant', constant_values=(0,0))
    
    else:
        # pad odd number of pix.
        return np.pad(feature_vector, (int(pad_amount/2 + 0.5), int(pad_amount/2 - 0.5)), 'constant', constant_values=(0,0))



    

def trim(feature_vector: np.ndarray, k: int, mode='random'):
    """
    Trim the feature_vector so 


    TODO: mode - smallest variance
    """
    pass
