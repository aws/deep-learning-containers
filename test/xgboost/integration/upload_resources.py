"""Upload integration test resources to S3.

Run this script once to populate s3://dlc-cicd-models/xgboost/integ_test_resources/
with the data needed by the integration tests.

The script expects the following data to already exist in the bucket
(from the original ai_algorithms_qa test infrastructure):
  - train/          (libsvm abalone training data)
  - test/           (libsvm abalone test data)
  - csv/train/      (csv training data)
  - csv/test/       (csv test data)
  - parquet/train/  (parquet training data)
  - parquet/test/   (parquet test data)
  - recordio-protobuf/train/  (protobuf training data)
  - recordio-protobuf/test/   (protobuf test data)
  - recordio-protobuf/sparse/train/
  - recordio-protobuf/sparse/test/
  - iris/train/     (iris csv training data)
  - iris/test/      (iris csv test data)
  - model_1.0/models/model.tar.gz  (pre-trained model for inference tests)
  - script_mode/code/abalone.1.2-1.tar.gz
  - script_mode/data/train/
  - script_mode/data/validation/
  - testdata/abalone_test.libsvm

This script packages and uploads the local script_mode resources.
"""

import os
import tarfile
import tempfile

import boto3

BUCKET = "dlc-cicd-models"
PREFIX = "xgboost/integ_test_resources"
RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")


def upload_script_mode_code():
    """Package abalone.py into a tar.gz and upload to S3."""
    s3 = boto3.client("s3")
    code_dir = os.path.join(RESOURCES_DIR, "script_mode", "code")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        with tarfile.open(tmp.name, "w:gz") as tar:
            tar.add(os.path.join(code_dir, "abalone.py"), arcname="abalone.py")
        s3_key = f"{PREFIX}/script_mode/code/abalone.1.2-1.tar.gz"
        s3.upload_file(tmp.name, BUCKET, s3_key)
        print(f"Uploaded s3://{BUCKET}/{s3_key}")
        os.unlink(tmp.name)


def upload_script_mode_data():
    """Upload local training/validation data for script mode tests."""
    s3 = boto3.client("s3")
    data_dir = os.path.join(RESOURCES_DIR, "script_mode", "data")

    for split in ("training", "validation"):
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split_dir} (not found)")
            continue
        # Map training -> train, validation -> validation in S3
        s3_split = "train" if split == "training" else "validation"
        for fname in os.listdir(split_dir):
            local_path = os.path.join(split_dir, fname)
            s3_key = f"{PREFIX}/script_mode/data/{s3_split}/{fname}"
            s3.upload_file(local_path, BUCKET, s3_key)
            print(f"Uploaded s3://{BUCKET}/{s3_key}")


if __name__ == "__main__":
    upload_script_mode_code()
    upload_script_mode_data()
    print("Done. Ensure other datasets are already in S3.")
