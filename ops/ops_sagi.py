from abc import abstractmethod
from typing import Hashable, List, Sequence, Optional
from fuse.data.utils.sample import get_sample_id
from fuse.utils import NDict
from fuse.data import OpBase
import numpy
import torch



class OpKeysToList(OpBase):
    """
    TODO
    """

    def __init__(self, keys: Optional[List[str]] = None, prefix: Optional[str] = None):
        """
        TODO
        """
        self._keys = keys
        self._prefix = prefix

        if keys == None and prefix == None:
            raise Exception("TODO")

        if keys != None and prefix != None:
            raise Exception("TODO")

    
    def __call__(self, sample_dict: NDict):

        res = []
        if self._keys:
            for key in self._keys:
                res.append(sample_dict[key])
        
        if self._prefix:
            pass


