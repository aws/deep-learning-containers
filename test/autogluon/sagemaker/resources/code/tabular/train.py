"""Train an AutoGluon tabular predictor and package its inference handler."""

import os
import shutil
from pathlib import Path

from autogluon.tabular import TabularDataset, TabularPredictor


def first_file(directory):
    return Path(directory) / min(os.listdir(directory))


if __name__ == "__main__":
    model_dir = Path(os.environ["SM_MODEL_DIR"])
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))

    if num_gpus:
        import torch

        assert torch.cuda.is_available(), "SM_NUM_GPUS > 0 but CUDA is unavailable"

    fit_args = {
        "hyperparameters": {
            "GBM": {"num_boost_round": 20},
            "NN_TORCH": {"num_epochs": 2},
        },
        "num_bag_folds": 3,
        "num_bag_sets": 1,
        "num_stack_levels": 0,
    }
    if num_gpus:
        fit_args["num_gpus"] = num_gpus

    predictor = TabularPredictor(
        label="class",
        eval_metric="roc_auc",
        path=str(model_dir),
    ).fit(
        TabularDataset(first_file(os.environ["SM_CHANNEL_TRAIN"])),
        **fit_args,
    )

    assert TabularPredictor.load(str(model_dir)).model_names(), "reloaded predictor has no models"

    resource_dir = Path(__file__).parent
    code_dir = model_dir / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy2(resource_dir / "serve.py", code_dir / "serve.py")
