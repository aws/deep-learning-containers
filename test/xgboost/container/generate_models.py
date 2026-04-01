#!/usr/bin/env python3
"""Generate XGBoost-compatible inference models and upload to S3.

Run inside the XGBoost container to ensure models match the container's version:

    docker run --rm -v ~/.aws:/root/.aws <IMAGE_URI> \
        python3 -c "$(cat test/xgboost/container/generate_models.py)"

Or mount and run directly:

    docker run --rm -v ~/.aws:/root/.aws -v $(pwd):/work -w /work <IMAGE_URI> \
        python3 test/xgboost/container/generate_models.py
"""

import os
import pickle
import tempfile

import boto3
import numpy as np
import xgboost as xgb

S3_BUCKET = "dlc-cicd-models"
S3_PREFIX = "xgboost/container_test_resources/inference/models"
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
    print(f"Downloading training data...")
    download_s3_dir(s3, S3_BUCKET, S3_TRAINING_PREFIX, data_dir)

    libsvm_dir = os.path.join(data_dir, "single-libsvm")
    csv_dir = os.path.join(data_dir, "single-csv")

    # --- mnist models (binary classification on agaricus) ---
    print("Generating mnist models...")
    dtrain = xgb.DMatrix(os.path.join(libsvm_dir, "agaricus.libsvm.train"))
    dtest = xgb.DMatrix(os.path.join(libsvm_dir, "agaricus.libsvm.test"))
    bst = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eval_metric": "error"},
                     dtrain, 10, evals=[(dtest, "test")])
    bst.save_model(os.path.join(out_dir, "mnist-xgb-model"))

    # --- diabetes binary classification ---
    print("Generating diabetes-binary-xgb-model...")
    bst_bin = xgb.train({"objective": "binary:hinge", "max_depth": 6}, dtrain, 10)
    bst_bin.save_model(os.path.join(out_dir, "diabetes-binary-xgb-model"))

    # --- insurance models ---
    print("Generating insurance models...")
    csv_train = np.genfromtxt(os.path.join(csv_dir, "train.csv"), delimiter=",")
    dtrain_csv = xgb.DMatrix(csv_train[:, 1:], label=csv_train[:, 0])
    bst_ins = xgb.train({"objective": "reg:squarederror", "max_depth": 6}, dtrain_csv, 10)
    bst_ins.save_model(os.path.join(out_dir, "insurance-xgb-model"))

    # --- Upload to S3 ---
    print(f"Uploading to s3://{S3_BUCKET}/{S3_PREFIX}/")
    for fname in os.listdir(out_dir):
        local = os.path.join(out_dir, fname)
        key = f"{S3_PREFIX}/{fname}"
        s3.upload_file(local, S3_BUCKET, key)
        print(f"  {fname} ({os.path.getsize(local)} bytes)")

    print(f"Done — models generated with XGBoost {xgb.__version__}")


if __name__ == "__main__":
    main()
