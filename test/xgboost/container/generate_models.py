#!/usr/bin/env python3
"""Generate XGBoost 3.0.5-compatible inference models and upload to S3.

Run this INSIDE the XGBoost container so models match the container's XGBoost version:

    docker run --rm -v ~/.aws:/root/.aws -v $(pwd):/work -w /work <IMAGE_URI> \
        python3 test/xgboost/container/generate_models.py

Models are saved locally to /tmp/xgb-models/ then uploaded to S3.
"""

import os
import pickle
import sys
import tempfile

import boto3
import numpy as np
import xgboost as xgb

S3_BUCKET = "dlc-cicd-models"
S3_PREFIX = "xgboost/container_test_resources/inference/models"

# We need the inference input data to know feature dimensions.
# Download from S3 to read the training data.
S3_TRAINING_PREFIX = "xgboost/container_test_resources/training/data"


def download_s3_dir(s3, bucket, prefix, local_dir):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = os.path.relpath(key, prefix)
            if rel == ".":
                continue
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, key, dest)


def main():
    out_dir = tempfile.mkdtemp(prefix="xgb-models-")
    data_dir = tempfile.mkdtemp(prefix="xgb-data-")
    s3 = boto3.client("s3")

    print(f"XGBoost version: {xgb.__version__}")
    print(f"Downloading training data to {data_dir}...")
    download_s3_dir(s3, S3_BUCKET, S3_TRAINING_PREFIX, data_dir)

    libsvm_dir = os.path.join(data_dir, "single-libsvm")
    csv_dir = os.path.join(data_dir, "single-csv")

    # --- mnist models (multiclass on agaricus as proxy — matches reference tests) ---
    print("Generating mnist models...")
    dtrain = xgb.DMatrix(os.path.join(libsvm_dir, "agaricus.libsvm.train"))
    dtest = xgb.DMatrix(os.path.join(libsvm_dir, "agaricus.libsvm.test"))
    params = {"objective": "binary:logistic", "max_depth": 6, "eval_metric": "error"}
    bst = xgb.train(params, dtrain, 10, evals=[(dtest, "test")])

    pkl_path = os.path.join(out_dir, "mnist-pkl-model")
    pickle.dump(bst, open(pkl_path, "wb"))
    print(f"  mnist-pkl-model: {os.path.getsize(pkl_path)} bytes")

    xgb_path = os.path.join(out_dir, "mnist-xgb-model")
    bst.save_model(xgb_path)
    print(f"  mnist-xgb-model: {os.path.getsize(xgb_path)} bytes")

    # --- diabetes binary classification model ---
    print("Generating diabetes-binary-xgb-model...")
    # Use agaricus data with binary:hinge to match reference test expectations
    params_bin = {"objective": "binary:hinge", "max_depth": 6}
    bst_bin = xgb.train(params_bin, dtrain, 10)
    bin_path = os.path.join(out_dir, "diabetes-binary-xgb-model")
    bst_bin.save_model(bin_path)
    print(f"  diabetes-binary-xgb-model: {os.path.getsize(bin_path)} bytes")

    # --- insurance models ---
    print("Generating insurance models...")
    # Load CSV training data (first column is label)
    csv_train = np.genfromtxt(os.path.join(csv_dir, "train.csv"), delimiter=",")
    dtrain_csv = xgb.DMatrix(csv_train[:, 1:], label=csv_train[:, 0])
    params_ins = {"objective": "reg:squarederror", "max_depth": 6}
    bst_ins = xgb.train(params_ins, dtrain_csv, 10)

    ins_pkl_path = os.path.join(out_dir, "insurance-pkl-model")
    pickle.dump(bst_ins, open(ins_pkl_path, "wb"))
    print(f"  insurance-pkl-model: {os.path.getsize(ins_pkl_path)} bytes")

    ins_xgb_path = os.path.join(out_dir, "insurance-xgb-model")
    bst_ins.save_model(ins_xgb_path)
    print(f"  insurance-xgb-model: {os.path.getsize(ins_xgb_path)} bytes")

    # --- salary model (single feature) ---
    print("Generating salary-pkl-model...")
    # salary data has single feature column, label is first column
    sal_pkl_path = os.path.join(out_dir, "salary-pkl-model")
    # Reuse insurance model since salary just needs a model that accepts single-column input
    # Train on a simple 1-feature dataset
    np.random.seed(42)
    X_sal = np.random.rand(100, 1)
    y_sal = X_sal[:, 0] * 50000 + np.random.randn(100) * 5000
    dtrain_sal = xgb.DMatrix(X_sal, label=y_sal)
    bst_sal = xgb.train({"objective": "reg:squarederror", "max_depth": 3}, dtrain_sal, 10)
    pickle.dump(bst_sal, open(sal_pkl_path, "wb"))
    print(f"  salary-pkl-model: {os.path.getsize(sal_pkl_path)} bytes")

    # --- Upload all models to S3 ---
    print(f"\nUploading models to s3://{S3_BUCKET}/{S3_PREFIX}/")
    for fname in os.listdir(out_dir):
        local = os.path.join(out_dir, fname)
        key = f"{S3_PREFIX}/{fname}"
        s3.upload_file(local, S3_BUCKET, key)
        print(f"  Uploaded {fname} ({os.path.getsize(local)} bytes)")

    print("\nDone! All models regenerated with XGBoost", xgb.__version__)


if __name__ == "__main__":
    main()
