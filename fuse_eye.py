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
from ops.ops_sagi import OpKeysToList, OpConvImageKernel, OpSubtractMean, OpExpandTensor
import skimage


class OpEYESampleIDDecode(OpBase):
    def __call__(self, sample_dict: NDict) -> NDict:
        """
        decodes sample id
        """

        sample_dict["data.sample_id_as_int"] = int(sample_dict["data.sample_id"])
        # Cast the sample ids from integers to strings to match fuse's sampler
        sample_dict["data.sample_id"] = str(sample_dict["data.sample_id"])
        return sample_dict


class EYE:
    """
    TODO
    """

    # bump whenever the static pipeline modified
    DATASET_VER = 0

    @staticmethod
    def download(data_path: str, sample_ids_to_download: Optional[Sequence[str]] = None) -> None:
        """
        TODO
        """
        pass

    @staticmethod
    def sample_ids(data_path: str) -> List[str]:
        """
        Gets the samples ids in trainset.
        """
        # data = arff.loadarff(data_path)
        # df = pd.DataFrame(data[0])

        samples = [i for i in range(10936)]
        return samples

    @staticmethod
    def static_pipeline(data_path: str) -> PipelineDefault:
        feature_columns = EYE.get_feature_columns()
        data = arff.loadarff(data_path)
        df = pd.DataFrame(data[0])
        base_image = skimage.data.shepp_logan_phantom()  # Temp

        static_pipeline = PipelineDefault(
            "static",
            [
                # Step 1: Decoding sample ID TODO delete (?)
                (OpEYESampleIDDecode(), dict()),
                # Step 2: load sample's features
                (
                    OpReadDataframe(
                        data=df,
                        key_column=None,
                        key_name="data.sample_id_as_int",
                        columns_to_extract=feature_columns,
                    ),
                    dict(prefix="data.feature"),
                ),
                # Step 2.5: delete feature to match k^2
                # OpFunc
                # Step 3: load all the features into a list
                (OpKeysToList(prefix="data.feature"), dict(key_out="data.vector")),
                (OpToNumpy(), dict(key="data.vector", dtype=float)),
                # Step 4: reshape to kerenl - shuki
                (OpReshapeVector(), dict(key_in_vector="data.vector", key_out="data.kernel")),
                # Step 4.1: subract mean
                (OpSubtractMean(), dict(key="data.kernel")),
                # Step 5: Convolve with base image - sagi
                (OpConvImageKernel(base_image=base_image), dict(key_in_kernel="data.kernel", key_out="data.input.img")),
                # Load label TODO
                (
                    OpReadDataframe(
                        data=df,
                        key_column=None,  # should be default None.. maybe fix in fuse
                        key_name="data.sample_id_as_int",
                        columns_to_extract=["label"],
                    ),
                    dict(prefix="data"),
                ),
                (OpToInt(), dict(key="data.label")),
                # DEBUG
                # (OpPrintShapes(num_samples=1), dict()),
                # (OpPrintTypes(num_samples=1), dict()),
                # (OpPrintKeysContent(num_samples=1), dict(keys=None)),
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
            (OpToTensor(), dict(key="data.input.img", dtype=torch.float)),
            (OpExpandTensor(), dict(key="data.input.img")),
            (OpPrintShapes(num_samples=1), dict()),
        ]

        return PipelineDefault("dynamic", dynamic_pipeline)

    @staticmethod
    def dataset(
        data_path: str,
        cache_path: str,
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

        if samples_ids is None:
            samples_ids = EYE.sample_ids(data_path)

        static_pipeline = EYE.static_pipeline(data_path)
        dynamic_pipeline = EYE.dynamic_pipeline(train, append=append_dyn_pipeline)

        # TODO: delete or reactivate
        cacher = SamplesCacher(
            f"eye_cache_ver{EYE.DATASET_VER}",
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

    def get_feature_columns() -> List[str]:

        list_of_columns = [
            # "lineNo",    -> id same as index
            "assgNo",
            "fixcount",
            "firstPassCnt",
            "P1stFixation",
            "P2stFixation",
            "prevFixDur",
            "firstfixDur",
            "firstPassFixDur",
            "nextFixDur",
            "firstSaccLen",
            "lastSaccLen",
            "prevFixPos",
            "landingPos",
            "leavingPos",
            "totalFixDur",
            "meanFixDur",
            "nRegressFrom",
            "regressLen",
            "nextWordRegress",
            "regressDur",
            "pupilDiamMax",
            "pupilDiamLag",
            "timePrtctg",
            "nWordsInTitle",
            # "titleNo",     -> temp for dim k^2
            "wordNo",
            # "label",    -> label column
        ]

        return list_of_columns


if __name__ == "__main__":
    run_local = True

    # switch to os.environ (?)
    if run_local:
        ROOT = "./_examples/eye"
        DATA_DIR = "./data/raw_data/eye_movements.arff"
    else:
        ROOT = "/tmp/_sagi/_examples/eye"
        DATA_DIR = "./sagi_dl_workshop/data/raw_data/eye_movements.arff"

    cache_dir = os.path.join(ROOT, "cache_dir")

    sp = EYE.static_pipeline(DATA_DIR)
    # print(sp)

    create_dir("./cacher")
    dataset = EYE.dataset(DATA_DIR, cache_dir, reset_cache=True, samples_ids=None, use_cacher=False)
    assert len(dataset) == 10936

    sample = dataset[0]
    # sample.print_tree()
    print("DONE!")
