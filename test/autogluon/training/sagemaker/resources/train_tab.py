"""SageMaker entry point: fit an AutoGluon TabularPredictor.

Channels: config (YAML with ag_predictor_args / ag_fit_args), train (CSV), test (CSV).
Asserts on real outcomes so a broken image fails the job, not just a crash.
"""

import argparse
import os

import yaml
from autogluon.tabular import TabularDataset, TabularPredictor


def first_file(channel_dir):
    return os.path.join(channel_dir, sorted(os.listdir(channel_dir))[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--n_gpus", default=os.environ.get("SM_NUM_GPUS"))
    parser.add_argument("--training_dir", default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test_dir", default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--ag_config", default=os.environ.get("SM_CHANNEL_CONFIG"))
    args, _ = parser.parse_known_args()
    os.makedirs(args.output_data_dir, exist_ok=True)

    with open(first_file(args.ag_config)) as f:
        config = yaml.safe_load(f)

    num_gpus = int(args.n_gpus or 0)
    if num_gpus:
        import torch

        assert torch.cuda.is_available(), "SM_NUM_GPUS > 0 but torch.cuda.is_available() is False"
        config["ag_fit_args"]["num_gpus"] = num_gpus

    train_data = TabularDataset(first_file(args.training_dir))
    predictor = TabularPredictor(path=args.model_dir, **config["ag_predictor_args"]).fit(
        train_data, **config["ag_fit_args"]
    )

    leaderboard = predictor.leaderboard()
    assert len(leaderboard) > 0, "fit produced no models"
    leaderboard.to_csv(os.path.join(args.output_data_dir, "leaderboard.csv"))

    if args.test_dir:
        test_data = TabularDataset(first_file(args.test_dir))
        predictor.predict_proba(test_data).to_csv(
            os.path.join(args.output_data_dir, "predictions.csv")
        )
        scores = predictor.evaluate(test_data)
        metric = predictor.eval_metric.name
        print(f"Evaluation: {scores}")
        assert scores[metric] > 0.5, f"{metric}={scores[metric]} is not better than chance"

    assert TabularPredictor.load(args.model_dir).model_names(), "reloaded predictor has no models"
