from random import sample
from typing import Hashable, List, Optional, Dict, Union
from fuse.utils.file_io.file_io import read_dataframe
import pandas as pd
import numpy as np
from fuse.data import OpBase
from fuse.utils.ndict import NDict
import numpy.typing as npt
import math
import numpy as np


class OpReshapeVector(OpBase):

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, **kwargs) -> Union[None, dict, List[dict]]:

        # reshape input to kernel of size k x k where k = sqrt(vec.shape[1])
        vec = sample_dict["data.vector"][:25]
        np_vec = np.array(vec)
        k = int(np.sqrt(np_vec.shape[0]))
        kernel = np_vec.reshape((k, k))
        sample_dict["data.kernel"] = kernel
        return sample_dict


class OpReshapeVectorV2(OpBase):
    """
    Reshape 1d vector with len k^2 to a 2d array with dims kxk

    Example of use:
        (OpReshapeVectorV2(), dict(key_in_vector="data.vector", key_out="data.kernel")),
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, key_in_vector: str, key_out: str) -> Union[None, dict, List[dict]]:

        # reshape input to kernel of size k x k where k = vec.shape[1]
        vec: np.ndarray = sample_dict[key_in_vector] # vec should be numpy array

        if vec.ndim != 1:
            raise Exception("Vec dimension should be 1")

        k = math.sqrt(vec.shape[0])

        if int(k) != k:
            raise Exception(f"Vector size square root should be an integer, but got {k}")
        
        k = int(k)
        res = vec.reshape((k, k))

        sample_dict[key_out] = res
        return sample_dict

