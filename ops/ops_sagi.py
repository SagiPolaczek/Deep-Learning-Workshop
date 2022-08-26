from abc import abstractmethod
from sys import prefix
from typing import Hashable, List, Sequence, Optional
from fuse.data.utils.sample import get_sample_id
from fuse.utils import NDict
from fuse.data import OpBase
import numpy as np
import torch
import scipy.signal as signal



class OpKeysToList(OpBase):
    """
    
    Example of use:
        (OpKeysToList(prefix="data.feature"), dict(key_out="data.vector")),


    """

    def __init__(self, keys: Optional[List[str]] = None, prefix: Optional[str] = None, delete_keys: bool = True):
        """
        TODO
        """
        super().__init__()
        self._keys = keys
        self._prefix = prefix
        self._delete_keys = delete_keys

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
            for key in sample_dict.keypaths():
                if key.startswith(self._prefix):
                    res.append(sample_dict[key])
                    if self._delete_keys:
                        del sample_dict[key]

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
        # print(f"DEBUG: unique image values = {np.unique(image)}")
        # print(f"DEBUG: image shape = {image.shape}")
        # print(f"DEBUG: image = {image}")
        return sample_dict
        

class OpSubtractMean(OpBase):
    """
    subtract mean from a numpy array
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, key: str):

        arr = sample_dict[key]
        mean = np.mean(arr)
        arr -= mean

        # print(f"DEBUG: arr mean = {mean}")
        sample_dict[key] = arr
        return sample_dict


class OpExpandTensor(OpBase):
    """
    Expand 2D Tensor into a 3D Tensor such that the first dim is empty
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, key: str):

        tensor = sample_dict[key]
        # tensor = tensor[None, :, :]
        tensor = torch.unsqueeze(tensor, dim=0)  # Same

        sample_dict[key] = tensor
        return sample_dict


class OpRenameKey(OpBase):
    """
    Rename key in the sample dict.

    Example of use:

    """

    def __call__(self, sample_dict: NDict, key_old: str, key_new: str, delete_old: bool = True):
        """
        
        :param key_old:
        :param key_new:
        :param delete_old:
        """

        sample_dict[key_new] = sample_dict[key_old]

        if delete_old:
            del sample_dict[key_old]

        return sample_dict

class OpEpsilonRenameLabel(OpBase):
    """
    rename labels: -1 -> 0
                    1 -> 1
    """
    def __call__(self, sample_dict: NDict, key: str) -> NDict:
        """
        :param key: key for label
        """
        label = sample_dict[key]
        if label == -1:
            sample_dict[key] = 0

        return sample_dict