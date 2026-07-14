#!/usr/bin/env python3
"""Regenerate MME sklearn model fixtures for a given sklearn version.

Fits two seeded RandomForestRegressor models on synthetic data matching the
shape used in test/sklearn/sagemaker/test_inference_mme.py (6 features, 1 output).
Emits two tarballs (mme_model_0.tar.gz, mme_model_1.tar.gz) suitable for uploading to
s3://amazonai-algorithms-integration-tests/input/scikit-learn/mme_models/<version>/.

Each tarball wraps a single joblib-serialized model at member name 'sklearn-model'
matching the layout the SageMaker sklearn container expects.

Usage:
    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install scikit-learn==<version> joblib numpy
    python regen_mme_fixtures.py --sklearn-version <version> --out-dir .

Prints the model's prediction on the CSV test payload from test_inference_mme.py
so the caller knows what the endpoint should return end-to-end.
"""

import argparse
import tarfile
import tempfile
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

N_SAMPLES = 100
N_FEATURES = 6
DATA_SEED_X = 42
DATA_SEED_Y = 43

# Matches SAMPLE_PAYLOAD in test_inference_mme.py — CSV with 2 rows, 6 features each.
TEST_PAYLOAD = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 4.0, 6.0, 9.0, 3.0],
    ]
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sklearn-version",
        required=True,
        help="Exact sklearn version to assert against sklearn.__version__ (e.g. 1.9.0)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory to write mme_model_{0,1}.tar.gz (default: current dir)",
    )
    args = parser.parse_args()

    if sklearn.__version__ != args.sklearn_version:
        raise SystemExit(
            f"Installed sklearn=={sklearn.__version__} does not match "
            f"--sklearn-version={args.sklearn_version}. "
            f"Install the exact version and re-run."
        )

    X = np.random.RandomState(DATA_SEED_X).rand(N_SAMPLES, N_FEATURES)
    y = np.random.RandomState(DATA_SEED_Y).rand(N_SAMPLES)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        model = RandomForestRegressor(random_state=i).fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "sklearn-model"
            joblib.dump(model, tmp_path)
            out_tar = args.out_dir / f"mme_model_{i}.tar.gz"
            with tarfile.open(out_tar, "w:gz") as tar:
                tar.add(tmp_path, arcname="sklearn-model")
        preds = model.predict(TEST_PAYLOAD)
        print(f"{out_tar} -> predict(TEST_PAYLOAD) = {preds.tolist()}")


if __name__ == "__main__":
    main()
