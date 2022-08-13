import os
from typing import OrderedDict
import logging
import copy


import torch
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from fuse.utils.utils_debug import FuseDebug
import fuse.utils.gpu as GPU
from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import create_dir, save_dataframe
from fuse.data.utils.split import dataset_balanced_division_to_folds

from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve

from fuse.data.datasets.caching.samples_cacher import SamplesCacher
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.dl.models.backbones.backbone_resnet import BackboneResnet
from fuse.dl.models.backbones.backbone_inception_resnet_v2 import BackboneInceptionResnetV2
from fuse.dl.models.heads.head_global_pooling_classifier import HeadGlobalPoolingClassifier

from fuse.dl.models import ModelMultiHead
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe

from fuse.dl.losses.loss_default import LossDefault
from fuse.eval.evaluator import EvaluatorDefault

from fuse_higgs import HIGGS
from ops.ops_sagi import *
from ops.ops_shaked import *

###########################################################################################################
# Fuse
###########################################################################################################

##########################################
# Debug modes
##########################################
mode = "debug"  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################
NUM_GPUS = 1
ROOT = "./_examples/higgs"
DATA_DIR = "/Users/shakedcaspi/Documents/tau/deep_learning_workshop/Deep-Learning-Workshop/data/raw_data/training.csv"
model_dir = os.path.join(ROOT, "model_dir")
PATHS = {
    "data_dir": DATA_DIR,
    "model_dir": model_dir,
    # "force_reset_model_dir": False,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
    "cache_dir": os.path.join(ROOT, "cache_dir"),
    "inference_dir": os.path.join(model_dir, "infer"),
    "eval_dir": os.path.join(model_dir, "eval"),
    "data_split_filename": os.path.join(ROOT, "higgs_split.pkl")
}

##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}
# ============
# Data
# ============
TRAIN_COMMON_PARAMS["data.batch_size"] = 8
TRAIN_COMMON_PARAMS["data.train_num_workers"] = 8
TRAIN_COMMON_PARAMS["data.validation_num_workers"] = 8
TRAIN_COMMON_PARAMS["data.cache_num_workers"] = 10
TRAIN_COMMON_PARAMS["data.num_folds"] = 5
TRAIN_COMMON_PARAMS["data.train_folds"] = [0, 1, 2]
TRAIN_COMMON_PARAMS["data.validation_folds"] = [3]
TRAIN_COMMON_PARAMS["data.samples_ids"] = None  # Use all data


# ===============
# PL Trainer
# ===============
TRAIN_COMMON_PARAMS["trainer.num_epochs"] = 10  # TODO raise
TRAIN_COMMON_PARAMS["trainer.num_devices"] = NUM_GPUS
TRAIN_COMMON_PARAMS["trainer.accelerator"] = "cpu"
TRAIN_COMMON_PARAMS["trainer.ckpt_path"] = None

# ===============
# Optimizer
# ===============
TRAIN_COMMON_PARAMS["opt.lr"] = 1e-4
TRAIN_COMMON_PARAMS["opt.weight_decay"] = 1e-3

# ===================================================================================================================
# Model
# ===================================================================================================================


def create_model() -> torch.nn.Module:

    model = ModelMultiHead(
        conv_inputs=(("data.input.img", 3),),
        backbone={
            "Resnet18": BackboneResnet(pretrained=True, in_channels=1, name="resnet18"),
            "InceptionResnetV2": BackboneInceptionResnetV2(input_channels_num=1, logical_units_num=43),
        }["InceptionResnetV2"],
        heads=[
            HeadGlobalPoolingClassifier(
                head_name="head_0",
                # dropout_rate=dropout_rate,
                # change if use resnet, i think to 512, need to double check
                conv_inputs=[("model.backbone_features", 1536)],
                num_classes=2,
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
    fuse_logger_start(
        output_path=paths["model_dir"], console_verbose_level=logging.INFO)

    print("Fuse Train")
    print(f'model_dir={paths["model_dir"]}')
    print(f'cache_dir={paths["cache_dir"]}')

    # ==============================================================================
    # Data
    # ==============================================================================

    # Train Data

    print("Train Data:")

    if mode == "debug":
        train_sample_ids = [
            "100000", "349997", "100009", "100008", "100007",  # class s
            "100001", "100002", "100003", "349996", "349995",  # class b
        ]
        validation_sample_ids = [
            "100036", "100037", "100038",  # class s
            "100033", "100034", "100035",  # class b
        ]

    else:

        # TODO - list your sample ids:
        # Fuse TIP - splitting the sample_ids to folds can be done by fuse.data.utils.split.dataset_balanced_division_to_folds().
        #            See (examples/fuse_examples/imaging/classification/stoic21/runner_stoic21.py)[../../examples/fuse_examples/imaging/classification/stoic21/runner_stoic21.py]
        all_dataset = HIGGS.dataset(
            paths["data_dir"],
            paths["cache_dir"],
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

    train_dataset = HIGGS.dataset(
        paths["data_dir"], paths["cache_dir"], reset_cache=True, samples_ids=train_sample_ids)

    # Create batch sampler
    print("- Create sampler:")
    sampler = None  # Debug, delete when finished
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.label",
        num_balanced_classes=2,
        batch_size=train_common_params["data.batch_size"],
    )

    print("- Create sampler: Done")

    # Create dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=train_common_params["data.train_num_workers"],
    )
    print("Train Data: Done")

    # Validation data
    print("Validation Data:")

    validation_dataset = HIGGS.dataset(
        paths["data_dir"], paths["cache_dir"], reset_cache=True, samples_ids=validation_sample_ids)

    # Create dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=train_common_params["data.batch_size"],
        num_workers=train_common_params["data.validation_num_workers"],
        collate_fn=CollateDefault(),
    )
    print("Validation Data: Done")

    # Create model
    print("Model:")
    model = create_model()
    print("Model: Done")

    # ==========================================================================================================================================
    #   Loss
    # ==========================================================================================================================================
    losses = {
        "cls_loss": LossDefault(pred="model.logits.head_0", target="data.label", callable=F.cross_entropy, weight=1.0),
    }

    # =========================================================================================================
    # Metrics - details can be found in (fuse/eval/README.md)[../../fuse/eval/README.md]
    #   1. Create seperately for train and validation (might be a deep copy, but not a shallow one).
    #   2. Set best_epoch_source:
    #       monitor: the metric name to track
    #       mode: either consider the "min" value to be best or the "max" value to be the best
    # =========================================================================================================
    class_names = ["s", "b"]
    train_metrics = OrderedDict(
        [
            # will apply argmax
            ("op", MetricApplyThresholds(pred="model.output.head_0")),
            ("auc", MetricAUCROC(pred="model.output.head_0",
             target="data.label", class_names=class_names)),
            ("accuracy", MetricAccuracy(
                pred="results:metrics.op.cls_pred", target="data.label")),
        ]
    )

    # use the same metrics in validation as well
    validation_metrics = copy.deepcopy(train_metrics)

    best_epoch_source = dict(
        monitor="validation.metrics.auc.macro_avg", mode="max")

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
    lr_sch_config = dict(scheduler=lr_scheduler,
                         monitor="validation.losses.total_loss")

    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(
        optimizer=optimizer, lr_scheduler=lr_sch_config)

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
    pl_trainer.fit(
        pl_module, train_dataloader, validation_dataloader, ckpt_path=train_common_params[
            "trainer.ckpt_path"]
    )

    print("Fuse Train: Done")


######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS["data.num_workers"] = TRAIN_COMMON_PARAMS["data.train_num_workers"]
INFER_COMMON_PARAMS["data.batch_size"] = 4
INFER_COMMON_PARAMS["infer_filename"] = os.path.join(
    PATHS["inference_dir"], "validation_set_infer.pickle")
# Fuse TIP: possible values are 'best', 'last' or epoch_index.
INFER_COMMON_PARAMS["checkpoint"] = "best"
INFER_COMMON_PARAMS["data.infer_folds"] = [4]  # infer validation set

######################################
# Inference Template
######################################


def run_infer(paths: dict, infer_common_params: dict) -> None:
    create_dir(paths["inference_dir"])
    infer_file = os.path.join(
        paths["inference_dir"], infer_common_params["infer_filename"])
    checkpoint_file = os.path.join(
        paths["model_dir"], infer_common_params["checkpoint"])

    # Logger
    fuse_logger_start(
        output_path=paths["inference_dir"], console_verbose_level=logging.INFO)
    print("Fuse Inference")
    print(f"infer_filename={infer_file}")

    # Data
    # assume exists and created in train func
    folds = load_pickle(paths["data_split_filename"])

    infer_sample_ids = []
    for fold in infer_common_params["data.infer_folds"]:
        infer_sample_ids += folds[fold]

    # Create dataset
    infer_dataset = HIGGS.dataset(
        paths["data_dir"], paths["cache_dir"], reset_cache=True, samples_ids=infer_sample_ids)

    # Create dataloader
    infer_dataloader = DataLoader(
        dataset=infer_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=infer_common_params["data.batch_size"],
        num_workers=infer_common_params["data.num_workers"],
        collate_fn=CollateDefault(),
    )

    # TODO - define / create a model
    model = ModelMultiHead(
        conv_inputs=(("data.input.input_0.tensor", 1),),
        backbone="TODO",  # Reference: BackboneInceptionResnetV2
        # References: HeadGlobalPoolingClassifier, HeadDenseSegmentation
        heads=["TODO"],
    )

    # load python lightning module
    pl_module = LightningModuleDefault.load_from_checkpoint(
        checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True
    )

    # set the prediction keys to extract and dump into file (the ones used be the evaluation function).
    pl_module.set_predictions_keys(
        [
            # TODO
        ]
    )

    # create a trainer instance and predict
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        accelerator=infer_common_params["trainer.accelerator"],
        devices=infer_common_params["trainer.num_devices"],
        auto_select_gpus=True,
    )
    predictions = pl_trainer.predict(
        pl_module, infer_dataloader, return_predictions=True)

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
    infer_file = os.path.join(
        paths["inference_dir"], eval_common_params["infer_filename"])

    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    lgr = logging.getLogger("Fuse")
    lgr.info("Fuse Eval", {"attrs": ["bold", "underline"]})

    # metrics
    metrics = OrderedDict(
        [
            # will apply argmax
            ("op", MetricApplyThresholds(pred="model.output.head_0")),
            ("auc", MetricAUCROC(pred="model.output.head_0", target="data.label")),
            ("accuracy", MetricAccuracy(
                pred="results:metrics.op.cls_pred", target="data.label")),
            (
                "roc",
                MetricROCCurve(
                    pred="model.output.head_0",
                    target="data.label",
                    output_filename=os.path.join(
                        paths["inference_dir"], "roc_curve.png"),
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
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    # GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    # Options: 'train', 'infer', 'eval'
    RUNNING_MODES = ["train", "infer", "eval"]

    # train
    if "train" in RUNNING_MODES:
        run_train(paths=PATHS, train_common_params=TRAIN_COMMON_PARAMS)

    # infer
    if "infer" in RUNNING_MODES:
        run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # eval
    if "eval" in RUNNING_MODES:
        run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)
