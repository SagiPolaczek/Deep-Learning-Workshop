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
from fuse.data.ops.ops_debug import OpPrintKeys, OpPrintKeysContent, OpPrintShapes
from fuseimg.data.ops.ops_debug import OpVis2DImage

from fuse.utils import NDict

from fuseimg.data.ops.image_loader import OpLoadImage
from fuseimg.data.ops.aug.color import OpAugColor, OpAugGaussian
from fuseimg.data.ops.aug.geometry import OpResizeTo, OpAugAffine2D
from fuse.utils.rand.param_sampler import Uniform, RandInt, RandBool


from ops.ops_sagi import *
from ops.ops_shaked import *
import skimage


def derive_label(sample_dict: NDict) -> NDict:
    """
    Takes the sample's ndict with the labels as key:value and assigns to sample_dict['data.label'] the index of the sample's class.
    Also delete all the labels' keys from sample_dict.

    for example:
        If the sample contains {'MEL': 0, 'NV': 1, 'BCC': 0, 'AK': 0, ... }
        will assign, sample_dict['data.label'] = 1 ('NV's index).
        Afterwards the sample_dict won't contain the class' names & values.
    """
    classes_names = ["s", "b"]

    label = 0
    for idx, cls_name in enumerate(classes_names):
        if int(sample_dict[f"data.cls_labels.{cls_name}"]) == 1:
            label = idx

    sample_dict["data.label"] = label
    return sample_dict


class HIGGS:
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
        data = pd.read_csv("./data/raw_data/training.csv")
        samples = list(range(data.shape[0]))
        return samples

    @staticmethod
    def static_pipeline(data_path: str) -> PipelineDefault:
        df = pd.read_csv("./data/raw_data/training.csv")
        df.drop(["EventId"], axis=1, inplace=True)
        # TODO: CHANGE THIS APPLY FUNC
        df["label"] = df["Label"].apply(lambda val: 1 if val == "s" else 0)
        feature_columns = HIGGS.get_feature_columns()

        base_image = skimage.data.shepp_logan_phantom()  # Temp

        static_pipeline = PipelineDefault(
            "static",
            [
                # Step 1: Decoding sample ID TODO delete (?)
                (OpHIGGSSampleIDDecode(), dict()),
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
        # return PipelineDefault("dynamic", [])

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
            samples_ids = HIGGS.sample_ids(data_path)

        static_pipeline = HIGGS.static_pipeline(data_path)
        dynamic_pipeline = HIGGS.dynamic_pipeline(train, append=append_dyn_pipeline)

        # TODO: delete or reactivate
        cacher = SamplesCacher(
            f"higgs_cache_ver{HIGGS.DATASET_VER}",
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
            # 'EventId',
            "DER_mass_MMC",
            "DER_mass_transverse_met_lep",
            "DER_mass_vis",
            "DER_pt_h",
            "DER_deltaeta_jet_jet",
            "DER_mass_jet_jet",
            "DER_prodeta_jet_jet",
            "DER_deltar_tau_lep",
            "DER_pt_tot",
            "DER_sum_pt",
            "DER_pt_ratio_lep_tau",
            "DER_met_phi_centrality",
            "DER_lep_eta_centrality",
            "PRI_tau_pt",
            "PRI_tau_eta",
            "PRI_tau_phi",
            "PRI_lep_pt",
            "PRI_lep_eta",
            "PRI_lep_phi",
            "PRI_met",
            "PRI_met_phi",
            "PRI_met_sumet",
            "PRI_jet_num",
            "PRI_jet_leading_pt",
            "PRI_jet_leading_eta",
            #  'PRI_jet_leading_phi',
            #  'PRI_jet_subleading_pt',
            #  'PRI_jet_subleading_eta',
            #  'PRI_jet_subleading_phi',
            #  'PRI_jet_all_pt',
            #  'Weight',
            #  'Label'
        ]

        return list_of_columns


if __name__ == "__main__":
    run_local = True

    # switch to os.environ (?)
    if run_local:
        ROOT = "./_examples/higgs"
        DATA_DIR = "./data/raw_data/training.csv"
    else:
        ROOT = "./_examples/eye"
        DATA_DIR = ".data/raw_data/training.csv"

    cache_dir = os.path.join(ROOT, "cache_dir")

    sp = HIGGS.static_pipeline(DATA_DIR)
    # print(sp)

    create_dir("./cacher")
    dataset = HIGGS.dataset(DATA_DIR, cache_dir, reset_cache=True, samples_ids=None, use_cacher=False)
    assert len(dataset) == 250000

    sample = dataset[0]
    # sample.print_tree()
    print("DONE!")
