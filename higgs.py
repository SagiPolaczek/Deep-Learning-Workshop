import os
from typing import Hashable, Optional, Sequence, List, Tuple
import torch
import pandas as pd
import numpy as np
import random

from fuse.data import DatasetDefault
from fuse.data.ops.ops_cast import OpToTensor, OpToNumpy, OpToInt
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.op_base import OpBase
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.ops_read import OpReadDataframe

from ops.custom_fuse_ops import *
import skimage


class HIGGS:
    """
    TODO
    """
    # bump whenever the static pipeline modified
    DATASET_VER = 0

    @staticmethod
    def sample_ids(train: bool = True) -> List[str]:
        """
        Gets the samples ids in trainset.
        """
        random.seed(42)
        if train:
            samples = [i for i in range(400000)]
            random.shuffle(samples)
            samples = samples[:10000]
        else:
            samples = [i for i in range(100000)]
            random.shuffle(samples)
            samples = samples[:2500]


        return samples

    @staticmethod
    def static_pipeline(data: pd.DataFrame, base_image: np.ndarray
                        ) -> PipelineDefault:

        feature_columns = list(data.columns)
        feature_columns.remove("0")
        label_column = ["0"]
        static_pipeline = PipelineDefault(
            "static",
            [

                (OpHIGGSSampleIDDecode(), dict()),
                # Step 1: load sample's features
                (
                    OpReadDataframe(
                        data=data,
                        key_column=None,
                        key_name="data.sample_id_as_int",
                        columns_to_extract=feature_columns,
                    ),
                    dict(prefix="data.feature"),
                ),
                # Step 2: load all the features into a list
                (OpKeysToList(prefix="data.feature"), dict(key_out="data.vector")),
                (OpToNumpy(), dict(key="data.vector", dtype=float)),
                # Step 3: reshape to kerenl - shuki
                (OpReshapeVector(), dict(
                    key_in_vector="data.vector", key_out="data.kernel")),
                # Step 4: subract mean
                (OpSubtractMean(), dict(key="data.kernel")),
                # Step 5: Convolve with base image - sagi
                (OpConvImageKernel(base_image=base_image), dict(
                    key_in_kernel="data.kernel", key_out="data.input.img")),
                # Load label
                (
                    OpReadDataframe(
                        data=data,
                        key_column=None,  # should be default None.. maybe fix in fuse
                        key_name="data.sample_id_as_int",
                        columns_to_extract=label_column,
                    ),
                    dict(prefix="data"),
                ),
                (OpRenameKey(), dict(key_old="data.0", key_new="data.label")),
                (OpToInt(), dict(key="data.label")),
                # DEBUG
                # (OpPrintShapes(num_samples=1), dict()),
                # (OpPrintTypes(num_samples=1), dict()),

                # (OpVis2DImage(num_samples=1), dict(
                # key="data.input.img", dtype="float")),
            ],
        )
        return static_pipeline

    @staticmethod
    def dynamic_pipeline(
        train: bool = False, append: Optional[Sequence[Tuple[OpBase, dict]]] = None
    ) -> PipelineDefault:

        dynamic_pipeline = [
            # Convert to tensor
            (OpToTensor(), dict(key="data.input.img", dtype=torch.float)),
            (OpExpandTensor(), dict(key="data.input.img")),
            # (OpPrintShapes(num_samples=1), dict()),
        ]

        return PipelineDefault("dynamic", dynamic_pipeline)
        # return PipelineDefault("dynamic", [])

    @staticmethod
    def dataset(
        cache_path: str,
        data: pd.DataFrame,
        base_image: np.ndarray,
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

        assert (data is not None and data_path is None) or (
            data is None and data_path is not None)

        if samples_ids is None:
            samples_ids = HIGGS.sample_ids(train)

        static_pipeline = HIGGS.static_pipeline(
            data=data, base_image=base_image)
        dynamic_pipeline = HIGGS.dynamic_pipeline()

        cacher = SamplesCacher(
            f"higgs_cache_ver{HIGGS.DATASET_VER}",
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

    if run_local:
        ROOT = "./_examples/higgs"
        DATA_DIR = ""
        samples_ids = [i for i in range(50)]
    else:
        ROOT = "/tmp/_shaked/_examples/higgs"
        DATA_DIR = ""
        samples_ids = None

    cache_dir = os.path.join(ROOT, "cache_dir")

    if debug:
        print("Loading debug data")
        train_data = pd.read_csv(
            "./data/raw_data/higgs/fs_debug_training_1000.csv"
        )
        test_data = pd.read_csv(
            "./data/raw_data/higgs/fs_debug_test_200.csv"
        )
        print("Done loading debug data!")

    else:
        train_data = pd.read_csv(
            "./data/raw_data/higgs/fs_training.csv"
        )
        test_data = pd.read_csv(
            "./data/raw_data/higgs/fs_test.csv"
        )
        print("Done loading data!")

    # Testing static pipeline initialization
    base_image = skimage.data.brick()

    sp = HIGGS.static_pipeline(data=train_data, base_image=base_image)

    dataset = HIGGS.dataset(
        data=train_data, base_image=base_image, cache_path=cache_dir, reset_cache=True, samples_ids=samples_ids)

    # all data pipeline will be executed
    sample = dataset[0]
    print("Done!")
