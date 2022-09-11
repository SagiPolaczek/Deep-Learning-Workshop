import math
from typing import List, Union, Optional
from fuse.utils import NDict
from fuse.data import OpBase
import numpy as np
import scipy.signal as signal
import torch



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
            raise Exception(f"Vector size square root should be an integer, but got {k}")

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

        sample_dict["data.sample_id_as_int"] = int(sample_dict["data.sample_id"])
        # Cast the sample ids from integers to strings to match fuse's sampler
        sample_dict["data.sample_id"] = str(sample_dict["data.sample_id"])
        return sample_dict


class OpBasicFeatureSelection(OpBase):
    def __call__(self, sample_dict: NDict) -> NDict:
        """
        decodes sample id
        """

        sample_dict["data.sample_id_as_int"] = int(sample_dict["data.sample_id"])
        # Cast the sample ids from integers to strings to match fuse's sampler
        sample_dict["data.sample_id"] = str(sample_dict["data.sample_id"])
        return sample_dict


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
    Expand tensor
    """

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, key: str, dim: int = 0):

        tensor = sample_dict[key]

        tensor = torch.unsqueeze(tensor, dim=dim)

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


class OpPadVecInOneSide(OpBase):
    """
    Pad vector in one side.

    [1,2,3] with padding=3 -> [1,2,3,0,0,0]

    as oppose to other pad funcs where we get [0,0,0,1,2,3,0,0,0]
    """

    def __call__(self, sample_dict: NDict, key_in: str, key_out: str, padding: int):

        vec: np.ndarray = sample_dict[key_in]

        padded_vec = np.pad(vec, (0, padding))

        sample_dict[key_out] = padded_vec
        return sample_dict
