import os
from typing import Hashable, Optional, Sequence, List
import torch
import pandas as pd
from catboost.datasets import epsilon

from fuse.data import DatasetDefault
from fuse.data.ops.ops_cast import OpToTensor, OpToNumpy, OpToInt
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.op_base import OpBase
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.utils import NDict

from ops.ops_shaked import OpReshapeVector
from ops.ops_sagi import OpKeysToList, OpExpandTensor, OpRenameKey, OpEpsilonRenameLabel, OpPadVecInOneSide


class OpEPSILONSampleIDDecode(OpBase):
    """
    decodes sample id
    """
    def __call__(self, sample_dict: NDict) -> NDict:
        sample_dict["data.sample_id_as_int"] = int(sample_dict["data.sample_id"])
        # Cast the sample ids from integers to strings to match fuse's sampler
        sample_dict["data.sample_id"] = str(sample_dict["data.sample_id"])
        return sample_dict


class EPSILON:
    """
    data.input.vector -> the raw input data as a vector
    data.input.sqr_vector -> raw input data as a square (orig + 25 padded zeros)
    data.label -> input label as 0 or 1

    """

    DATASET_VER = 0

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

                # Step 2: load sample's features
                (OpReadDataframe(
                        data=data,
                        key_column = None,
                        key_name = "data.sample_id_as_int",
                        columns_to_extract=feature_columns,
                    ),
                    dict(prefix="data.feature")),

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
            ],
        )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline() -> PipelineDefault:
        """
        TODO
        """

        dynamic_pipeline = [
            # Step 1 - Pad and reshape to 2D matrix
            (OpPadVecInOneSide(), dict(key_in="data.input.vector", key_out="data.input.vector_padded", padding=25)),
            (OpReshapeVector(), dict(key_in_vector="data.input.vector_padded", key_out="data.input.sqr_vector")),

            # Step 2 - Convert to tensors
            (OpToTensor(), dict(key="data.input.vector", dtype=torch.float)),
            (OpToTensor(), dict(key="data.input.sqr_vector", dtype=torch.float)),

            # Step 3 - Exapnd to match model dims
            (OpExpandTensor(), dict(key="data.input.sqr_vector")),
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
        samples_ids: Optional[Sequence[Hashable]] = None,
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
            samples_ids = EPSILON.sample_ids(train)

        static_pipeline = EPSILON.static_pipeline(data=data)
        dynamic_pipeline = EPSILON.dynamic_pipeline()

        # TODO: delete or reactivate
        cacher = SamplesCacher(
            f"eye_cache_ver{EPSILON.DATASET_VER}",
            static_pipeline,
            [cache_path],
            restart_cache=reset_cache,
            workers=num_workers,
        )
        
        my_dataset = DatasetDefault(
            sample_ids=samples_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
            cacher=cacher,
        )

        my_dataset.create()
        return my_dataset


if __name__ == "__main__":
    # Main script for testing data pipelines
    run_local = True
    debug = True

    # switch to os.environ (?)
    if run_local:
        ROOT = "./_examples/epsilon"
        DATA_DIR = ""
        samples_ids = [i for i in range(1000)]
    else:
        ROOT = "/tmp/_sagi/_examples/epsilon"
        DATA_DIR=""
        samples_ids = None

    cache_dir = os.path.join(ROOT, "cache_dir")

    if debug:
        print("Loading debug data")
        train_data = pd.read_csv("/Users/sagipolaczek/Documents/Studies/git-repos/DLW/data/raw_data/eps/train_debug_1000.csv")
        test_data = pd.read_csv("/Users/sagipolaczek/Documents/Studies/git-repos/DLW/data/raw_data/eps/test_debug_200.csv")
        print("Done loading debug data!")

    else:
        print("Downloading data...")
        train_data, test_data = epsilon()
        print("Done downloading data!")

    # Testing static pipeline initialization
    sp = EPSILON.static_pipeline(data=train_data)

    dataset = EPSILON.dataset(
        data=train_data, cache_path=cache_dir, reset_cache=True, samples_ids=samples_ids
    )

    # all data pipeline will be executed
    sample = dataset[0]
    print("Done!")