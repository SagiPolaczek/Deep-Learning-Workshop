from sklearn.datasets import load_iris
import os
from zipfile import ZipFile
from fuse.utils.file_io.file_io import create_dir, read_dataframe
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
from fuse.data.ops.ops_debug import OpPrintKeys, OpPrintKeysContent, OpPrintShapes, OpPrintTypes
from fuseimg.data.ops.ops_debug import OpVis2DImage

from fuse.utils import NDict

from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.aug.color import OpAugColor, OpAugGaussian
from fuseimg.data.ops.aug.geometry import OpResizeTo, OpAugAffine2D
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from ops.ops_sagi import *
from ops.ops_shaked import *
import skimage


def feature_selection(data: pd.DataFrame, k: int) -> List[str]:
    X = data.drop("0", axis=1)
    y = data["0"]
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X, y)
    return selector.get_feature_names_out()


if __name__ == "__main__":
    # Main script for testing data pipelines

    train_data_1 = pd.read_csv(
        "./data/raw_data/higgs/training_part1.csv")
    train_data_2 = pd.read_csv(
        "./data/raw_data/higgs/training_part2.csv")
    train_data_3 = pd.read_csv(
        "./data/raw_data/higgs/training_part3.csv")

    train_data = pd.concat(
        [train_data_1, train_data_2, train_data_3], axis=0, ignore_index=True)

    test_data = pd.read_csv(
        "./data/raw_data/higgs/test.csv",
    )

    train_data.columns = [str(col) for col in train_data.columns]
    test_data.columns = [str(col) for col in test_data.columns]

    # preprocessing phase

    features_list_without_label = list(
        feature_selection(train_data, 25))

    features_list_with_label = features_list_without_label + ["0"]

    train_data = train_data[features_list_with_label].copy()
    test_data = test_data[features_list_with_label].copy()

    train_data.iloc[:50].to_csv(
        "/Users/shakedcaspi/Documents/tau/deep_learning_workshop/Deep-Learning-Workshop/data/raw_data/higgs/fs_debug_training_1000.csv", index=False)
    test_data.iloc[:20].to_csv(
        "/Users/shakedcaspi/Documents/tau/deep_learning_workshop/Deep-Learning-Workshop/data/raw_data/higgs/fs_debug_test_200.csv", index=False)

    train_data.to_csv(
        "/Users/shakedcaspi/Documents/tau/deep_learning_workshop/Deep-Learning-Workshop/data/raw_data/higgs/fs_training.csv", index=False)
    test_data.to_csv(
        "/Users/shakedcaspi/Documents/tau/deep_learning_workshop/Deep-Learning-Workshop/data/raw_data/higgs/fs_test.csv", index=False)
