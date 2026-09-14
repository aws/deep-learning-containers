"""Train an AutoGluon time series predictor and package its inference handler."""

import os
import shutil
from pathlib import Path

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

PREDICTION_LENGTH = 5


def first_file(directory):
    return Path(directory) / min(os.listdir(directory))


if __name__ == "__main__":
    model_dir = Path(os.environ["SM_MODEL_DIR"])
    num_gpus = int(os.environ.get("SM_NUM_GPUS", "0"))

    if num_gpus:
        import torch

        assert torch.cuda.is_available(), "SM_NUM_GPUS > 0 but CUDA is unavailable"

    train_data = TimeSeriesDataFrame.from_data_frame(
        pd.read_csv(first_file(os.environ["SM_CHANNEL_TRAIN"])),
        id_column="item_id",
        timestamp_column="timestamp",
    )
    predictor = TimeSeriesPredictor(
        prediction_length=PREDICTION_LENGTH,
        target="target",
        path=str(model_dir),
    ).fit(
        train_data,
        hyperparameters={
            "DeepAR": {
                "batch_size": 4,
                "hidden_size": 8,
                "max_epochs": 2,
                "num_batches_per_epoch": 2,
                "num_layers": 1,
                "trainer_kwargs": {
                    "accelerator": "gpu" if num_gpus else "cpu",
                    "devices": 1,
                },
            },
            "SeasonalNaive": {"seasonal_period": 6},
        },
        enable_ensemble=False,
    )

    reloaded = TimeSeriesPredictor.load(str(model_dir))
    assert {"DeepAR", "SeasonalNaive"} <= set(reloaded.model_names())

    resource_dir = Path(__file__).parent
    code_dir = model_dir / "code"
    code_dir.mkdir(exist_ok=True)
    shutil.copy2(resource_dir / "serve.py", code_dir / "serve.py")
