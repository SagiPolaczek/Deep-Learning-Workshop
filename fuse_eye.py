import os
from zipfile import ZipFile
from fuse.utils.file_io.file_io import create_dir
import wget
from typing import Hashable, Optional, Sequence, List, Tuple
import torch
from scipy.io import arff
import pandas as pd

from fuse.data import DatasetDefault
from fuse.data.ops.ops_cast import OpToTensor
from fuse.data.utils.sample import get_sample_id
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.ops.op_base import OpBase
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_common import OpLambda, OpOverrideNaN
from fuseimg.data.ops.color import OpToRange
from fuse.data.ops.ops_debug import OpPrintKeys, OpPrintKeysContent

from fuse.utils import NDict

from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.aug.color import OpAugColor, OpAugGaussian
from fuseimg.data.ops.aug.geometry import OpResizeTo, OpAugAffine2D
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool


class OpEYESampleIDDecode(OpBase):
    def __call__(self, sample_dict: NDict) -> NDict:
        """
        decodes sample id
        """

        return sample_dict


def derive_label(sample_dict: NDict) -> NDict:
    """
    Takes the sample's ndict with the labels as key:value and assigns to sample_dict['data.label'] the index of the sample's class.
    Also delete all the labels' keys from sample_dict.

    for example:
        If the sample contains {'MEL': 0, 'NV': 1, 'BCC': 0, 'AK': 0, ... }
        will assign, sample_dict['data.label'] = 1 ('NV's index).
        Afterwards the sample_dict won't contain the class' names & values.
    """
    classes_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    label = 0
    for idx, cls_name in enumerate(classes_names):
        if int(sample_dict[f"data.cls_labels.{cls_name}"]) == 1:
            label = idx

    sample_dict["data.label"] = label
    return sample_dict


class EYE:
    """
    TODO
    """

    # bump whenever the static pipeline modified
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
    def sample_ids(data_path: str) -> List[str]:
        """
        Gets the samples ids in trainset.
        #"""
        # data = arff.loadarff(data_path)
        # df = pd.DataFrame(data[0])

        samples = [i for i in range(10936)]
        return samples

    @staticmethod
    def static_pipeline(data_path: str) -> PipelineDefault:
        feature_columns = EYE.get_feature_columns()
        data = arff.loadarff(data_path)
        df = pd.DataFrame(data[0])


        static_pipeline = PipelineDefault(
            "static",
            [
                # Step 1: Decoding sample ID TODO delete (?)
                (OpEYESampleIDDecode(), dict()),

                # Step 2: load sample's features
                (OpReadDataframe(
                        data=df,
                        key_column = None,
                        columns_to_extract=feature_columns,
                    ),
                    dict(prefix="data.feature")),
                
                

                (OpPrintKeysContent(num_samples=1), dict(keys=None)),


                # Load label
                # (OpLambda(func=derive_label), dict()),
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
            # Resize images to 300x300x3
            (
                OpResizeTo(channels_first=True),
                dict(
                    key="data.input.img",
                    output_shape=(300, 300, 3),
                    mode="reflect",
                    anti_aliasing=True,
                ),
            ),
            # Convert to tensor for the augmentation process
            (OpToTensor(), dict(key="data.input.img", dtype=torch.float)),
        ]

        if train:
            dynamic_pipeline += [
                # Augmentation
                (
                    OpSample(OpAugAffine2D()),
                    dict(
                        key="data.input.img",
                        rotate=Uniform(-180.0, 180.0),
                        scale=Uniform(0.9, 1.1),
                        flip=(RandBool(0.3), RandBool(0.3)),
                        translate=(RandInt(-50, 50), RandInt(-50, 50)),
                    ),
                ),
                # Color augmentation
                (
                    OpSample(OpAugColor()),
                    dict(
                        key="data.input.img",
                        gamma=Uniform(0.9, 1.1),
                        contrast=Uniform(0.85, 1.15),
                        add=Uniform(-0.06, 0.06),
                        mul=Uniform(0.95, 1.05),
                    ),
                ),
                # Add Gaussian noise
                (OpAugGaussian(), dict(key="data.input.img", std=0.03)),
            ]

        if append is not None:
            dynamic_pipeline += append

        # return PipelineDefault("dynamic", dynamic_pipeline)
        return PipelineDefault("dynamic", [])

    @staticmethod
    def dataset(
        data_path: str,
        cache_path: str,
        train: bool = False,
        reset_cache: bool = False,
        num_workers: int = 10,
        append_dyn_pipeline: Optional[Sequence[Tuple[OpBase, dict]]] = None,
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

        if samples_ids is None:
            samples_ids = EYE.sample_ids(data_path)

        static_pipeline = EYE.static_pipeline(data_path)
        dynamic_pipeline = EYE.dynamic_pipeline(train, append=append_dyn_pipeline)

        # TODO: delete or reactivate
        # cacher = SamplesCacher(
        #     f"eye_cache_ver{EYE.DATASET_VER}",
        #     static_pipeline,
        #     [cache_path],
        #     restart_cache=reset_cache,
        #     workers=num_workers,
        # )

        my_dataset = DatasetDefault(
            sample_ids=samples_ids,
            static_pipeline=static_pipeline,
            dynamic_pipeline=dynamic_pipeline,
            cacher=None,
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
            "titleNo",
            "wordNo",
            # "label",    -> label column
        ]
        return list_of_columns


if __name__ == "__main__":
    ROOT = "./test_dataset"
    cache_dir = os.path.join(ROOT, "cache_dir")

    data_dir = "/Users/sagipolaczek/Documents/Studies/git-repos/DLW/data/raw_data/eye_movements.arff"

    sp = EYE.static_pipeline(data_dir)
    # print(sp)

    create_dir("./cacher")
    dataset = EYE.dataset(
        data_dir, cache_dir, reset_cache=True, samples_ids=None
    )
    assert len(dataset) == 10936

    for sample_index in range(10):
        sample = dataset[sample_index]
        assert get_sample_id(sample) == (sample_index +1)
