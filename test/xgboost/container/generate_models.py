#!/usr/bin/env python3
"""Generate XGBoost 3.0.5-compatible inference models and upload to S3.

Trains models using the inference input data so feature dimensions match.
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

    # --- mnist-xgb-model (784 features, multiclass) ---
    # Train on mnist-700.csv which has the same feature dimensions as all mnist-*.csv/libsvm/pbr inputs
    print("Generating mnist-xgb-model...")
    mnist_csv = np.genfromtxt(os.path.join(input_dir, "mnist-700.csv"), delimiter=",")
    # First column is label for mnist CSV inference data
    # Generate synthetic labels for training (actual values don't matter for inference shape tests)
    n_rows, n_cols = mnist_csv.shape
    labels = np.random.randint(0, 10, size=n_rows).astype(float)
    dtrain_mnist = xgb.DMatrix(mnist_csv, label=labels)
    print(f"  mnist training matrix: {n_rows} rows, {n_cols} cols")
    bst_mnist = xgb.train(
        {"objective": "multi:softmax", "num_class": 10, "max_depth": 6},
        dtrain_mnist, 10,
    )
    bst_mnist.save_model(os.path.join(out_dir, "mnist-xgb-model"))

    # --- diabetes-binary-xgb-model (same feature dims as diabetes_inference.csv) ---
    print("Generating diabetes-binary-xgb-model...")
    diabetes_csv = np.genfromtxt(os.path.join(input_dir, "diabetes_inference.csv"), delimiter=",")
    n_rows_d, n_cols_d = diabetes_csv.shape
    labels_d = np.random.randint(0, 2, size=n_rows_d).astype(float)
    dtrain_diabetes = xgb.DMatrix(diabetes_csv, label=labels_d)
    print(f"  diabetes training matrix: {n_rows_d} rows, {n_cols_d} cols")
    bst_diabetes = xgb.train(
        {"objective": "binary:hinge", "max_depth": 6},
        dtrain_diabetes, 10,
    )
    bst_diabetes.save_model(os.path.join(out_dir, "diabetes-binary-xgb-model"))

    # --- insurance-xgb-model (trained on actual CSV training data) ---
    print("Generating insurance-xgb-model...")
    csv_dir = os.path.join(train_dir, "single-csv")
    csv_train = np.genfromtxt(os.path.join(csv_dir, "train.csv"), delimiter=",")
    dtrain_ins = xgb.DMatrix(csv_train[:, 1:], label=csv_train[:, 0])
    print(f"  insurance training matrix: {csv_train.shape[0]} rows, {csv_train.shape[1] - 1} cols")
    bst_ins = xgb.train(
        {"objective": "reg:squarederror", "max_depth": 6},
        dtrain_ins, 10,
    )
    bst_ins.save_model(os.path.join(out_dir, "insurance-xgb-model"))

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
