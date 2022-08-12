from random import sample
from typing import Hashable, List, Optional, Dict, Union
from fuse.utils.file_io.file_io import read_dataframe
import pandas as pd

from fuse.data import OpBase
from fuse.utils.ndict import NDict
import numpy.typing as npt


class OpReshapeVector(OpBase):

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, **kwargs) -> Union[None, dict, List[dict]]:

        # reshape input to kernel of size k x k where k = vec.shape[1]
        vec = sample_dict["data.values_vector"]
        k = self.vec.shape[1]
        self.kernel = self.vec.reshape((k, k))
        sample_dict["data.kernel"] = self.kernel
        return super().__call__(sample_dict, **kwargs)
