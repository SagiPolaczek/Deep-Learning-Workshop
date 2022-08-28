from abc import abstractmethod
import math
from sys import prefix
from typing import Hashable, List, Sequence, Optional, Union
from fuse.data.utils.sample import get_sample_id
from fuse.utils import NDict
from fuse.data import OpBase
import numpy as np
import torch
import scipy.signal as signal


class OpReshapeVector(OpBase):
    """
    Reshape 1d vector with len k^2 to a 2d array with dims kxk

    Example of use:
        (OpReshapeVectorV2(), dict(key_in_vector="data.vector", key_out="data.kernel")),
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, key_in_vector: str, key_out: str) -> Union[None, dict, List[dict]]:

        # reshape input to kernel of size k x k where k = vec.shape[1]
        # vec should be numpy array
        vec: np.ndarray = sample_dict[key_in_vector]

        if vec.ndim != 1:
            raise Exception("Vec dimension should be 1")

        k = math.sqrt(vec.shape[0])

        if int(k) != k:
            raise Exception(
                f"Vector size square root should be an integer, but got {k}")

        k = int(k)
        res = vec.reshape((k, k))

        sample_dict[key_out] = res
        return sample_dict


class OpHIGGSSampleIDDecode(OpBase):
    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict) -> NDict:
        """
        decodes sample id
        """

        sample_dict["data.sample_id_as_int"] = int(
            sample_dict["data.sample_id"])
        # Cast the sample ids from integers to strings to match fuse's sampler
        sample_dict["data.sample_id"] = str(sample_dict["data.sample_id"])
        return sample_dict


class OpBasicFeatureSelection(OpBase):
    def __call__(self, sample_dict: NDict) -> NDict:
        """
        decodes sample id
        """

        sample_dict["data.sample_id_as_int"] = int(
            sample_dict["data.sample_id"])
        # Cast the sample ids from integers to strings to match fuse's sampler
        sample_dict["data.sample_id"] = str(sample_dict["data.sample_id"])
        return sample_dict

