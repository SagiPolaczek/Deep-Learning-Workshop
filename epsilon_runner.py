"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

"""

import os
from typing import OrderedDict
import logging
import copy
import pandas as pd

import torch
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from fuse.utils.utils_debug import FuseDebug
import fuse.utils.gpu as GPU
from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import create_dir, save_dataframe
from fuse.data.utils.split import dataset_balanced_division_to_folds

from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.dl.models.backbones.backbone_mlp import BackboneMultilayerPerceptron
from fuse.dl.models.heads.head_global_pooling_classifier import HeadGlobalPoolingClassifier
from fuse.dl.models.heads.head_generic import HeadGeneric
from fuse.dl.models.heads.common import ClassifierMLP

from fuse.dl.models import ModelMultiHead
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe

from fuse.dl.losses.loss_default import LossDefault
from fuse.eval.evaluator import EvaluatorDefault

from fuse_epsilon import EPSILON
from autoencoder import Encoder, Decoder, OurEncodingLoss
import torchvision.models as models

###########################################################################################################
# Fuse
###########################################################################################################

##########################################
# Experiments
##########################################
run_local = True  # set 'False' if running remote
experiment = "MLP"  # Choose from supported experiments

supported_experiments = [
    "MLP",  # TODO elaborate
    "full",
    "disjoint",  # TODO elaborate
    "overlap",  # TODO elaborate
]

assert experiment in supported_experiments, f"runner doesn't support experiment ({experiment})."

##########################################
# Debug modes
##########################################
mode = "default"  # switch to "debug" in a debug session
debug = FuseDebug(mode)

##########################################
# Paths
##########################################
NUM_GPUS = 1

# TODO switch to os.environ (?)
ROOT = "./_examples/epsilon"
if run_local:
    train_data_path = "/Users/sagipolaczek/Documents/Studies/git-repos/DLW/data/raw_data/eps/train_debug_1000.csv"
    eval_data_path = "/Users/sagipolaczek/Documents/Studies/git-repos/DLW/data/raw_data/eps/test_debug_200.csv"
else:
    train_data_path = "./fuse_workshop/_examples/epsilon/data/train_data.csv"
    eval_data_path = "./fuse_workshop/_examples/epsilon/data/test_data.csv"


model_dir = os.path.join(ROOT, f"model_dir_{experiment}")
PATHS = {
    "model_dir": model_dir,
    "cache_dir": os.path.join(ROOT, "cache_dir"),
    "inference_dir": os.path.join(model_dir, "infer"),
    "eval_dir": os.path.join(model_dir, "eval"),
    "data_split_filename": os.path.join(ROOT, "eps_split.pkl"),
}


##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}
# ============
# Data
# ============
TRAIN_COMMON_PARAMS["data.batch_size"] = 64
TRAIN_COMMON_PARAMS["data.train_num_workers"] = 10
TRAIN_COMMON_PARAMS["data.validation_num_workers"] = 10
TRAIN_COMMON_PARAMS["data.cache_num_workers"] = 10
TRAIN_COMMON_PARAMS["data.num_folds"] = 5
TRAIN_COMMON_PARAMS["data.train_folds"] = [0, 1, 2, 3]
TRAIN_COMMON_PARAMS["data.validation_folds"] = [4]
TRAIN_COMMON_PARAMS["data.samples_ids"] = [i for i in range(1000)] if run_local else None


# ===============
# PL Trainer
# ===============
TRAIN_COMMON_PARAMS["trainer.num_epochs"] = 1 if run_local else 15
TRAIN_COMMON_PARAMS["trainer.num_devices"] = NUM_GPUS
TRAIN_COMMON_PARAMS["trainer.accelerator"] = "cpu" if run_local else "gpu"

# ===============
# Optimizer
# ===============
TRAIN_COMMON_PARAMS["opt.lr"] = 1e-4
TRAIN_COMMON_PARAMS["opt.weight_decay"] = 1e-3

# ===================================================================================================================
# Model
# ===================================================================================================================


def create_model(experiment: str) -> torch.nn.Module:
    """
    TODO elaborate
    :param experiment:
    """
    if experiment == "MLP":
        model = ModelMultiHead(
            conv_inputs=(("data.input.vector", 1),),
            backbone=BackboneMultilayerPerceptron(mlp_input_size=2000),
            heads=[
                HeadGlobalPoolingClassifier(
                    head_name="head_cls",
                    # dropout_rate=dropout_rate,
                    conv_inputs=[("model.backbone_features", 384)],
                    shared_classifier_head=ClassifierMLP(
                        in_ch=384, num_classes=2, layers_description=(256,), dropout_rate=0.1
                    ),
                    pooling="avg",
                ),
            ],
        )

    else:  # Experiments envolve data autoencoding
        encoding_channels = 3

        model = ModelMultiHead(
            conv_inputs=(("data.input.sqr_vector", 1),),
            backbone=Encoder(in_channels=1, out_channels=encoding_channels, verbose=True),  # Encoder
            key_out_features="data.encoding",
            heads=[
                # Decoder
                HeadGeneric(
                    head_name="head_decoder",
                    conv_inputs=[("data.encoding", encoding_channels)],
                    head=Decoder(in_channels=encoding_channels, out_channels=1, verbose=True),
                ),
                # ResNet
                HeadGeneric(
                    head_name="head_resnet",
                    conv_inputs=[("data.encoding", encoding_channels)],
                    head=models.resnet50(pretrained=False, progress=True),
                ),
                # Classifier
                HeadGlobalPoolingClassifier(
                    head_name="head_cls",
                    # dropout_rate=dropout_rate,
                    conv_inputs=[
                        ("model.head_resnet", 1000)
                    ],  # change if use resnet, I think to 512, need to double check
                    shared_classifier_head=ClassifierMLP(
                        in_ch=1000, num_classes=2, layers_description=(256,), dropout_rate=0.1
                    ),
                    pooling="avg",
                ),
            ],
        )

    return model


#################################
# Train Template
#################################
def run_train(paths: dict, train_common_params: dict) -> None:
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)

    if run_local:
        print("Run LOCAL")

    else:
        print("Run REMOTE")

    print("Fuse Train")
    print(f'model_dir={paths["model_dir"]}')
    print(f'cache_dir={paths["cache_dir"]}')

    # ==============================================================================
    # Data
    # ==============================================================================

    #### Train Data

    print("Train Data:")
    print("Loading data...")
    TRAIN_DATA = pd.read_csv(train_data_path)
    print("Loading data - Done!")

    ### Split into train and validation
    all_dataset = EPSILON.dataset(
        paths["cache_dir"],
        data=TRAIN_DATA,
        train=True,
        reset_cache=False,
        num_workers=train_common_params["data.train_num_workers"],
        samples_ids=train_common_params["data.samples_ids"],
    )

    folds = dataset_balanced_division_to_folds(
        dataset=all_dataset,
        output_split_filename=paths["data_split_filename"],
        keys_to_balance=["data.label"],
        nfolds=train_common_params["data.num_folds"],
    )

    train_sample_ids = []
    for fold in train_common_params["data.train_folds"]:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in train_common_params["data.validation_folds"]:
        validation_sample_ids += folds[fold]

    train_dataset = EPSILON.dataset(
        paths["cache_dir"], data=TRAIN_DATA, reset_cache=False, samples_ids=train_sample_ids, train=True
    )

    ## Create batch sampler
    print("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.label",
        num_balanced_classes=2,
        batch_size=train_common_params["data.batch_size"],
    )

    print("- Create sampler: Done")

    ## Create dataloader
    print("- Create train dataloader:")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=train_common_params["data.train_num_workers"],
    )
    print("- Create train dataloader: Done")
    print("Train Data: Done")

    #### Validation data
    print("Validation Data:")

    validation_dataset = EPSILON.dataset(
        paths["cache_dir"], data=TRAIN_DATA, reset_cache=False, samples_ids=validation_sample_ids
    )

    ## Create dataloader
    print("- Create validation dataloader:")
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=train_common_params["data.batch_size"],
        num_workers=train_common_params["data.validation_num_workers"],
        collate_fn=CollateDefault(),
    )
    print("- Create validation dataloader: Done")
    print("Validation Data: Done")

    ## Create model
    print("Model:")
    model = create_model(experiment=experiment)
    print("Model: Done")

    # ==========================================================================================================================================
    #   Loss
    #   TODO Elaborate
    # ==========================================================================================================================================
    losses = {
        "cls_loss": LossDefault(pred="model.logits.head_cls", target="data.label", callable=F.cross_entropy, weight=1.0)
    }

    if experiment != "MLP":
        losses["ae_loss"] = LossDefault(pred="model.head_decoder", target="data.input.sqr_vector", callable=F.mse_loss, weight=1.0)
        losses["encoding_loss"] = OurEncodingLoss(key_encoding="data.encoding", mode=experiment, weight=100.0)

    # =========================================================================================================
    # Metrics
    # =========================================================================================================
    class_names = ["CLASS_0", "CLASS_1"]
    train_metrics = OrderedDict(
        [
            ("op", MetricApplyThresholds(pred="model.output.head_cls")),  # will apply argmax
            ("auc", MetricAUCROC(pred="model.output.head_cls", target="data.label", class_names=class_names)),
            ("accuracy", MetricAccuracy(pred="results:metrics.op.cls_pred", target="data.label")),
        ]
    )

    validation_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    best_epoch_source = dict(monitor="validation.metrics.auc.macro_avg", mode="max")

    # =====================================================================================
    #  Train - using PyTorch Lightning
    #  Create training objects, PL module and PL trainer.
    # =====================================================================================
    print("Fuse Train:")

    # create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_common_params["opt.lr"],
        weight_decay=train_common_params["opt.weight_decay"],
    )

    # create scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")

    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=paths["model_dir"],
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )

    # create lightining trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train_common_params["trainer.num_epochs"],
        accelerator=train_common_params["trainer.accelerator"],
        devices=train_common_params["trainer.num_devices"],
        auto_select_gpus=True,
    )

    # train
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader)

    print("Fuse Train: Done")


######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS["data.num_workers"] = TRAIN_COMMON_PARAMS["data.train_num_workers"]
INFER_COMMON_PARAMS["data.batch_size"] = TRAIN_COMMON_PARAMS["data.batch_size"]
INFER_COMMON_PARAMS["infer_filename"] = os.path.join(PATHS["inference_dir"], "validation_set_infer.pickle")
INFER_COMMON_PARAMS["checkpoint"] = "best_epoch.ckpt"  # Fuse TIP: possible values are 'best', 'last' or epoch_index.
INFER_COMMON_PARAMS["data.samples_ids"] = [i for i in range(200)] if run_local else None
INFER_COMMON_PARAMS["trainer.num_devices"] = TRAIN_COMMON_PARAMS["trainer.num_devices"]
INFER_COMMON_PARAMS["trainer.accelerator"] = TRAIN_COMMON_PARAMS["trainer.accelerator"]

######################################
# Inference Template
######################################


def run_infer(paths: dict, infer_common_params: dict) -> None:
    create_dir(paths["inference_dir"])
    infer_file = INFER_COMMON_PARAMS["infer_filename"]
    checkpoint_file = os.path.join(paths["model_dir"], infer_common_params["checkpoint"])

    ## Logger
    fuse_logger_start(output_path=paths["inference_dir"], console_verbose_level=logging.INFO)
    print("Fuse Inference")
    print(f"infer_filename={infer_file}")

    # Create dataset
    print("Loading data...")
    INFER_DATA = pd.read_csv(eval_data_path)
    print("Loading data - Done!")

    infer_dataset = EPSILON.dataset(
        paths["cache_dir"],
        data=INFER_DATA,
        reset_cache=False,
        train=False,
        samples_ids=infer_common_params["data.samples_ids"],
    )

    ## Create dataloader
    infer_dataloader = DataLoader(
        dataset=infer_dataset,
        batch_size=infer_common_params["data.batch_size"],
        num_workers=infer_common_params["data.num_workers"],
        collate_fn=CollateDefault(),
    )

    model = create_model(experiment=experiment)

    # load python lightning module
    pl_module = LightningModuleDefault.load_from_checkpoint(
        checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True
    )

    # set the prediction keys to extract and dump into file (the ones used be the evaluation function).
    pl_module.set_predictions_keys(["model.output.head_cls", "data.label"])

    # create a trainer instance and predict
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        accelerator=infer_common_params["trainer.accelerator"],
        devices=infer_common_params["trainer.num_devices"],
        auto_select_gpus=True,
    )
    predictions = pl_trainer.predict(pl_module, infer_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)

    print("Fuse Inference: Done")


######################################
# Eval Template
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS["infer_filename"] = INFER_COMMON_PARAMS["infer_filename"]


def run_eval(paths: dict, eval_common_params: dict) -> None:

    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Eval", {"attrs": ["bold", "underline"]})

    infer_file = eval_common_params["infer_filename"]

    # metrics
    metrics = OrderedDict(
        [
            ("op", MetricApplyThresholds(pred="model.output.head_cls")),  # will apply argmax
            ("auc", MetricAUCROC(pred="model.output.head_cls", target="data.label")),
            ("accuracy", MetricAccuracy(pred="results:metrics.op.cls_pred", target="data.label")),
            (
                "roc",
                MetricROCCurve(
                    pred="model.output.head_cls",
                    target="data.label",
                    output_filename=os.path.join(paths["inference_dir"], "roc_curve.png"),
                ),
            ),
        ]
    )

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(
        ids=None,
        data=infer_file,
        metrics=metrics,
        output_dir=paths["eval_dir"],
    )

    print("Fuse Eval: Done")
    return results


######################################
# Run
######################################
if __name__ == "__main__":
    if not run_local:
        # uncomment if you want to use specific gpus instead of automatically looking for free ones
        force_gpus = None  # [0]
        GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    RUNNING_MODES = ["train", "infer", "eval"]  # Options: 'train', 'infer', 'eval'

    # train
    if "train" in RUNNING_MODES:
        run_train(paths=PATHS, train_common_params=TRAIN_COMMON_PARAMS)

    # infer
    if "infer" in RUNNING_MODES:
        run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # eval
    if "eval" in RUNNING_MODES:
        run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)
