from abc import abstractmethod
from sys import prefix
from typing import Hashable, List, Sequence, Optional
from fuse.data.utils.sample import get_sample_id
from fuse.utils import NDict
from fuse.data import OpBase
import numpy
import torch
import scipy.signal as signal



class OpKeysToList(OpBase):
    """
    
    Example of use:
        (OpKeysToList(prefix="data.feature"), dict(key_out="data.vector")),


    """

    def __init__(self, keys: Optional[List[str]] = None, prefix: Optional[str] = None):
        """
        TODO
        """
        super().__init__()
        self._keys = keys
        self._prefix = prefix

        if keys == None and prefix == None:
            raise Exception("TODO")

        if keys != None and prefix != None:
            raise Exception("TODO")

    
    def __call__(self, sample_dict: NDict, key_out: str):
        """
        TODO
        """

        res = []

        if self._keys:
            for key in self._keys:
                res.append(sample_dict[key])
        
        elif self._prefix:
            print("HERE3")
            for key in sample_dict.keypaths():
                print("HERE2")
                if key.startswith(self._prefix):
                    print("HERE1")
                    res.append(sample_dict[key])

        sample_dict[key_out] = res
        return sample_dict



class OpConvImageKernel(OpBase):
    """
    convolve image with a given kernel
    """

    def __init__(self, base_image):
        super().__init__()
        self._base_image = base_image

    def __call__(self, sample_dict: NDict, key_in_kernel: str, key_out: str) -> NDict:
        
        kernel = sample_dict[key_in_kernel]
        image = signal.convolve2d(self._base_image, kernel)

        sample_dict[key_out] = image
        return sample_dict