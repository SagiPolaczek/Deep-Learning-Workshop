from zipfile import ZipFile
from fuse import data
from fuse.data.ops.ops_debug import OpPrintKeys, OpPrintKeysContent
from fuse.utils.file_io.file_io import create_dir
from typing import Hashable, Optional, Sequence, List, Tuple
import torch
import os
from fuse.data import DatasetDefault
from fuse.data.ops.ops_cast import OpToTensor
from fuse.data.utils.sample import get_sample_id
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.utils import NDict
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool

from fuse.data.ops.op_base import OpBase
from fuse.data.ops.ops_aug_common import OpSample
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.ops.ops_common import OpLambda
from fuseimg.data.ops.aug.geometry import OpResizeTo, OpAugAffine2D
from fuseimg.data.ops.aug.color import OpAugColor, OpAugGaussian
# --- added ops by me
from ops.ops_shaked import OpReshapeVector
# --- added ops by sagi
from ops.ops_sagi import OpKeysToList


import pandas as pd


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


class HIGGS:
    """
        HIGGS challenge to classify an event to background ot signal.
        an event refers to the results just after a fundamental interaction takes place between
        subatomic particles, occurring in a very short time span, at a well-localized region of space.
        A background event is explained by the existing theories and previous observations.
        A signal event indicates a process that cannot be described by previous observations
        and leads to the potential discovery of a new particle.
    """
    # bump whenever the static pipeline modified
    DATASET_VER = 0
    DATA_PATH = "/Users/shakedcaspi/Documents/tau/deep_learning_workshop/Deep-Learning-Workshop/data/raw_data/training.csv"
    CLASS_NAMES = ["b", "s"]

    # TODO:
    @staticmethod
    def download(data_path: str, sample_ids_to_download: Optional[Sequence[str]] = None) -> None:
        pass

    @staticmethod
    def sample_ids(data_path: str) -> List[str]:
        """
        Gets the samples ids in trainset.
        """
        data = pd.read_csv(data_path)
        return data["EventId"]

    @staticmethod
    def static_pipeline(data_path: str) -> PipelineDefault:

        data = pd.read_csv(data_path)
        feature_columns = data.columns.drop(["Weight", "Label"])

        rename_cls_labels = {
            c: f"data.cls_labels.{c}" for c in HIGGS.CLASS_NAMES}
        # also extract image (sample_id)
        rename_cls_labels["sample"] = "data.cls_labels.sample_id"

        static_pipeline = PipelineDefault(
            "static",
            [   # Read Data Frame
                (OpReadDataframe(data,
                                 key_column="EventId",
                                 columns_to_extract=feature_columns),
                 dict(prefix="data.feature")),
                (OpKeysToList(prefix="data.feature"), dict(key_out="data.vector")),
                (OpPrintKeysContent(num_samples=1), dict()),
                (OpReshapeVector(), dict()),
            ],
        )
        return static_pipeline

    @ staticmethod
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
                dict(key="data.input.img",
                     output_shape=(300, 300, 3),
                     mode="reflect",
                     anti_aliasing=True),
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
                (OpAugGaussian(),
                    dict(key="data.input.img",
                         std=0.03)),
            ]

        if append is not None:
            dynamic_pipeline += append

        return PipelineDefault("dynamic", dynamic_pipeline)

    @ staticmethod
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
        # HIGGS.download(data_path=data_path, sample_ids_to_download=samples_ids)

        if samples_ids is None:
            samples_ids = HIGGS.sample_ids(data_path)

        static_pipeline = HIGGS.static_pipeline(data_path)
        # dynamic_pipeline = HIGGS.dynamic_pipeline(
        #     train, append=append_dyn_pipeline)
        dynamic_pipeline = None

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
            cacher=None
        )

        my_dataset.create()
        return my_dataset

    @staticmethod
    def get_feature_columns(data_path: str) -> List[str]:
        """
        Gets the samples ids in trainset.
        """
        data = pd.read_csv(data_path)
        features_cols = data.columns.drop(
            ["EventId", "Weight", "Label"]).to_list()
        return features_cols


if __name__ == "__main__":
    data_path = "/Users/shakedcaspi/Documents/tau/deep_learning_workshop/Deep-Learning-Workshop/data/raw_data/training.csv"
    ROOT = "./test_dataset"
    cache_dir = os.path.join(ROOT, "cache_dir")

    sp = HIGGS.static_pipeline(data_path)

    create_dir("./cacher")
    dataset = HIGGS.dataset(
        data_path, cache_dir, reset_cache=True, samples_ids=None
    )

    ids = dataset.get_all_sample_ids()
    sample = dataset.getitem(100000)
    sample.print_tree()
