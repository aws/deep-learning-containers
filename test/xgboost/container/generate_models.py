#!/usr/bin/env python3
"""Generate XGBoost 3.0.5-compatible inference models and upload to S3.

Uses inference input data to create models with matching feature dimensions.
This is valid for container tests — we're testing the container's ability to
load models and serve predictions, not model accuracy.

Run on CI host with: pip install xgboost==3.0.5 boto3 numpy
"""

import os
import tempfile

import boto3
import numpy as np
import xgboost as xgb

S3_BUCKET = "dlc-cicd-models"
S3_PREFIX = "xgboost/container_test_resources/inference/models"
S3_INPUT_PREFIX = "xgboost/container_test_resources/inference/input"
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
    input_dir = tempfile.mkdtemp(prefix="xgb-input-")
    train_dir = tempfile.mkdtemp(prefix="xgb-train-")
    s3 = boto3.client("s3")

    print(f"XGBoost version: {xgb.__version__}")
    print("Downloading inference input data...")
    download_s3_dir(s3, S3_BUCKET, S3_INPUT_PREFIX, input_dir)
    print("Downloading training data...")
    download_s3_dir(s3, S3_BUCKET, S3_TRAINING_PREFIX, train_dir)

    # --- mnist-xgb-model ---
    # mnist-700.csv: first column is label, remaining are features
    # libsvm files use 1-based indexing with max index 785, so set num_feature=785
    # to ensure model accepts all inference input formats
    print("Generating mnist-xgb-model...")
    mnist_data = np.genfromtxt(os.path.join(input_dir, "mnist-700.csv"), delimiter=",")
    labels = mnist_data[:, 0]
    features = mnist_data[:, 1:]
    n_features = 785  # max feature index in libsvm files
    # Pad features to n_features if needed
    if features.shape[1] < n_features:
        pad = np.zeros((features.shape[0], n_features - features.shape[1]))
        features = np.concatenate([features, pad], axis=1)
    dtrain = xgb.DMatrix(features, label=labels)
    bst = xgb.train({"objective": "multi:softmax", "num_class": 10, "max_depth": 6},
                     dtrain, 10)
    bst.save_model(os.path.join(out_dir, "mnist-xgb-model"))
    print(f"  {features.shape[0]} rows x {features.shape[1]} features")

    # --- diabetes-binary-xgb-model ---
    print("Generating diabetes-binary-xgb-model...")
    diabetes_data = np.genfromtxt(os.path.join(input_dir, "diabetes_inference.csv"), delimiter=",")
    labels_d = np.random.randint(0, 2, size=diabetes_data.shape[0]).astype(float)
    dtrain_d = xgb.DMatrix(diabetes_data, label=labels_d)
    bst_d = xgb.train({"objective": "binary:hinge", "max_depth": 6}, dtrain_d, 10)
    bst_d.save_model(os.path.join(out_dir, "diabetes-binary-xgb-model"))
    print(f"  {diabetes_data.shape[0]} rows x {diabetes_data.shape[1]} cols")

    # --- insurance-xgb-model (from actual training CSV) ---
    print("Generating insurance-xgb-model...")
    csv_train = np.genfromtxt(os.path.join(train_dir, "single-csv", "train.csv"), delimiter=",")
    dtrain_ins = xgb.DMatrix(csv_train[:, 1:], label=csv_train[:, 0])
    bst_ins = xgb.train({"objective": "reg:squarederror", "max_depth": 6}, dtrain_ins, 10)
    bst_ins.save_model(os.path.join(out_dir, "insurance-xgb-model"))
    print(f"  {csv_train.shape[0]} rows x {csv_train.shape[1] - 1} cols")

    # --- Upload to S3 ---
    print(f"\nUploading to s3://{S3_BUCKET}/{S3_PREFIX}/")
    for fname in sorted(os.listdir(out_dir)):
        local = os.path.join(out_dir, fname)
        key = f"{S3_PREFIX}/{fname}"
        s3.upload_file(local, S3_BUCKET, key)
        print(f"  {fname} ({os.path.getsize(local)} bytes)")

    print(f"\nDone — models generated with XGBoost {xgb.__version__}")


if __name__ == "__main__":
    main()
