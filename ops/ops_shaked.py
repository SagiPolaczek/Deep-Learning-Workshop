from random import sample
from typing import Hashable, List, Optional, Dict, Union
from fuse.utils.file_io.file_io import read_dataframe
import pandas as pd
import numpy as np
from fuse.data import OpBase
from fuse.utils.ndict import NDict
import numpy.typing as npt


class OpReshapeVector(OpBase):

    def __init__(self):
        super().__init__()

    def __call__(self, sample_dict: NDict, **kwargs) -> Union[None, dict, List[dict]]:

        # reshape input to kernel of size k x k where k = sqrt(vec.shape[1])
        vec = sample_dict["data.vector"]
        np_vec = np.array(vec)
        k = int(np.sqrt(np_vec.shape[0]))
        kernel = np_vec.reshape((k, k))
        sample_dict["data.kernel"] = kernel
        return super().__call__(sample_dict, **kwargs)
