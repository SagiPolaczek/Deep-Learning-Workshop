import os
from zipfile import ZipFile
from fuse.utils.file_io.file_io import create_dir
import wget
from typing import Hashable, Optional, Sequence, List, Tuple
import torch
from scipy.io import arff
import pandas as pd
import numpy as np

from fuse.data import DatasetDefault
from fuse.data.ops.ops_cast import OpToTensor, OpToNumpy, OpToInt
from fuse.data.utils.sample import get_sample_id
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.op_base import OpBase
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_common import OpLambda, OpOverrideNaN
from fuseimg.data.ops.color import OpToRange, OpNormalizeAgainstSelf
from fuse.data.ops.ops_debug import OpPrintKeys, OpPrintKeysContent, OpPrintTypes, OpPrintShapes
from fuseimg.data.ops.ops_debug import OpVis2DImage

from fuse.utils import NDict

from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.aug.color import OpAugColor, OpAugGaussian
from fuseimg.data.ops.aug.geometry import OpResizeTo, OpAugAffine2D, OpAugUnsqueeze3DFrom2D
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool

from ops.ops_shaked import OpReshapeVector
from ops.ops_sagi import OpKeysToList, OpConvImageKernel, OpSubtractMean, OpExpandTensor, OpRenameKey, OpEpsilonRenameLabel
import skimage

from catboost.datasets import epsilon


class OpEPSILONSampleIDDecode(OpBase):
    def __call__(self, sample_dict: NDict) -> NDict:
        """
        decodes sample id
        """

        sample_dict["data.sample_id_as_int"] = int(sample_dict["data.sample_id"])
        # Cast the sample ids from integers to strings to match fuse's sampler
        sample_dict["data.sample_id"] = str(sample_dict["data.sample_id"])
        return sample_dict



class EPSILON:

    DATASET_VER = 0

    @staticmethod
    def download(
        data_path: str, sample_ids_to_download: Optional[Sequence[str]] = None
    ) -> None:
        """
        TODO
        """
        pass


    @staticmethod
    def sample_ids(train: bool = True) -> List[str]:
        """
        Gets the samples ids in trainset.
        """
        if train:
            samples = [i for i in range(400000)]
        else:
            samples = [i for i in range(100000)]
        return samples

    @staticmethod
    def static_pipeline(data: pd.DataFrame) -> PipelineDefault:
        """
        :param data: a table such that the first column is the label, and all the other 2000 are features
        """
        # needs to be str if data is loaded via 'read_csv'
        feature_columns = [ str(_) for _ in range(1, 2001)]  # All 2000 features. 
        label_column = ["0"]

        static_pipeline = PipelineDefault(
            "static",
            [
                # Step 1: Decoding sample ID TODO delete (?)
                (OpEPSILONSampleIDDecode(), dict()),
                (OpPrintKeysContent(num_samples=1), dict(keys=None)),

                # Step 2: load sample's features
                (OpReadDataframe(
                        data=data,
                        key_column = None,
                        key_name = "data.sample_id_as_int",
                        columns_to_extract=feature_columns,
                    ),
                    dict(prefix="data.feature")),

                # Step 2.5: delete feature to match k^2
                # OpFunc

                # Step 3: load all the features into a numpy array
                (OpKeysToList(prefix="data.feature"), dict(key_out="data.input.vector")),
                (OpToNumpy(), dict(key="data.input.vector", dtype=float)),

                # Step 4: Load label 
                (OpReadDataframe(
                        data=data,
                        key_column = None,  # should be default None.. maybe fix in fuse
                        key_name = "data.sample_id_as_int",
                        columns_to_extract=label_column,
                    ),
                    dict(prefix="data")),
                
                (OpRenameKey(), dict(key_old="data.0", key_new="data.label")),
                (OpEpsilonRenameLabel(), dict(key="data.label")),

                (OpToInt(), dict(key="data.label")),
                # DEBUG
                # (OpPrintShapes(num_samples=1), dict()),
                # (OpPrintTypes(num_samples=1), dict()),
                (OpPrintKeysContent(num_samples=1), dict(keys=None)),
                # (OpVis2DImage(), dict(key="data.input.img", dtype="float")),

            ],
        )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(
        train: bool = False, append: Optional[Sequence[Tuple[OpBase, dict]]] = None
    ) -> PipelineDefault:
        """
        Get suggested dynamic pipeline. including pre-processing that might be modified and augmentation operations.
        :param train: add augmentation if True
        :param append: pipeline steps to append at the end of the suggested pipeline
        """

        dynamic_pipeline = [
            # Convert to tensor
            (OpToTensor(), dict(key="data.input.vector", dtype=torch.float)),
            # (OpExpandTensor(), dict(key="data.input.vector")),
            # (OpExpandTensor(), dict(key="data.input.vector")),
            # (OpPrintShapes(num_samples=1), dict()),
        ]

        return PipelineDefault("dynamic", dynamic_pipeline)


    @staticmethod
    def dataset(
        cache_path: str,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[str] = None,
        train: bool = False,
        reset_cache: bool = False,
        num_workers: int = 10,
        append_dyn_pipeline: Optional[Sequence[Tuple[OpBase, dict]]] = None,
        samples_ids: Optional[Sequence[Hashable]] = None,
        use_cacher: bool = True,
    ) -> DatasetDefault:
        """
        Get cached dataset
        :param train: if true returns the train dataset, else the validation one.
        :param reset_cache: set to True to reset the cache
        :param num_workers: number of processes used for caching
        :param append_dyn_pipeline: pipeline steps to append at the end of the suggested dynamic pipeline
        :param sample_ids: dataset including the specified sample_ids or None for all the samples.
        """
        # Download data if doesn't exist
        # TODO (?)

        assert (data is not None and data_path is None) or (data is None and data_path is not None)
        
        if data_path:
            # read data
            data = None # TODO
            pass

        if samples_ids is None:
            samples_ids = EPSILON.sample_ids()

        static_pipeline = EPSILON.static_pipeline(data=data)
        dynamic_pipeline = EPSILON.dynamic_pipeline(train, append=append_dyn_pipeline)

        # TODO: delete or reactivate
        cacher = SamplesCacher(
            f"eye_cache_ver{EPSILON.DATASET_VER}",
            static_pipeline,
            [cache_path],
            restart_cache=reset_cache,
            workers=num_workers,
        )

        if not use_cacher:  # debugging
            cacher = None
        
        my_dataset = DatasetDefault(
            sample_ids=samples_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
            cacher=cacher,
        )

        my_dataset.create()
        return my_dataset


if __name__ == "__main__":
    run_local = True

    # switch to os.environ (?)
    if run_local:
        ROOT = "./_examples/epsilon"
        DATA_DIR = ""
    else:
        ROOT = "/tmp/_sagi/_examples/epsilon"
        DATA_DIR=""

    cache_dir = os.path.join(ROOT, "cache_dir")

    debug = True
    if debug:
        print("Loading debug data")
        train_data = pd.read_csv("/Users/sagipolaczek/Documents/Studies/git-repos/DLW/data/raw_data/eps/train_debug_1000.csv")
        test_data = pd.read_csv("/Users/sagipolaczek/Documents/Studies/git-repos/DLW/data/raw_data/eps/test_debug_200.csv")
        print("Done loading debug data!")

    else:
        print("Downloading data...")
        train_data, test_data = epsilon()
        print("Done downloading data!")

    # Testing sp initialization
    sp = EPSILON.static_pipeline(data=train_data)

    dataset = EPSILON.dataset(
        data=train_data, cache_path=cache_dir, reset_cache=True, samples_ids=None, use_cacher=False
    )

    sample = dataset[0]
    print("Done!")