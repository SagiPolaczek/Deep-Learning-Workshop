from epsilon_runner import *

"""

----------------------------
If you wish to 'play' with the demo/runner, please change the following values in the 'epsilon_runner.py' source file.
For changes in the hyper parameters consider changing: TRAIN_COMMON_PARAMS, INFER_COMMON_PARAMS, EVAL_COMMON_PARAMS.

For further modifications please check "fuse-med-ml" documentation and source code at: https://github.com/BiomedSciAI/fuse-med-ml,
or contact us.

----------------------------



experiment = "full"  # Choose from supported experiments

supported_experiments = [
    "MLP",
    "full",
    "disjoint",
    "overlap",
]

# Set you paths

ROOT = "./_examples/epsilon" 
model_dir = os.path.join(ROOT, f"model_dir_{experiment}")
PATHS = {
    "model_dir": model_dir,
    "cache_dir_train": os.path.join(ROOT, f"cache_dir_train"),
    "cache_dir_eval": os.path.join(ROOT, f"cache_dir_eval"),
    "inference_dir": os.path.join(model_dir, "infer"),
    "eval_dir": os.path.join(model_dir, "eval"),
    "data_split_filename": os.path.join(ROOT, "eps_split.pkl"),
}

"""


if __name__ == "__main__":

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
