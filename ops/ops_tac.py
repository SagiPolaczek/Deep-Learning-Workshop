from typing import Hashable, List, Optional, Dict, Union
from fuse.utils.file_io.file_io import read_dataframe
import pandas as pd

from fuse.data import OpBase
from fuse.utils.ndict import NDict


class OpGenerateKernel(OpBase):

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        data_filename: Optional[str] = None,
        columns_to_extract: Optional[List[str]] = None,
        deleted_feature_names: Optional[str] = None,
    ):
        super().__init__()

        # store input
        self._data_filename = data_filename
        self._columns_to_extract = columns_to_extract
        self.columns_to_extract = columns_to_extract
        self.deleted_feature_names
        df = data

        # verify input
        if data is None and data_filename is None:
            msg = "Error: need to provide either in-memory DataFrame or a path to file."
            raise Exception(msg)
        elif data is not None and data_filename is not None:
            msg = "Error: need to provide either 'data' or 'data_filename' args, bot not both."
            raise Exception(msg)

        # read dataframe
        if self._data_filename is not None:
            df = read_dataframe(self._data_filename)

        # extract only specified columns (in case not specified, extract all)
        if self._columns_to_extract is not None:
            df = df[self._columns_to_extract]

        # delete features if needed to make the feature be a number with a integer root
        if self.number_of_deleted_features:
            self.df.drop(self.deleted_feature_names, axis=1, inplace=True)

        # reshape input to kernel of size k x k
        k = self.df.shape[1]
        df.reshape((k, k))
