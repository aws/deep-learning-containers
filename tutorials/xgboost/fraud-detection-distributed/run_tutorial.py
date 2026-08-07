"""
End-to-end orchestration for Distributed Fraud Detection with XGBoost + Dask.

Generates synthetic fraud data, partitions it for Dask, trains on SageMaker
using distributed multi-GPU XGBoost (Algorithm mode), deploys, and tests.
"""

import argparse
import os
import time

import boto3
import numpy as np
import pandas as pd
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# GPU counts per instance type for partition calculation
GPUS_PER_INSTANCE = {
    "ml.g5.xlarge": 1,
    "ml.g5.2xlarge": 1,
    "ml.g5.4xlarge": 1,
    "ml.g5.8xlarge": 1,
    "ml.g5.12xlarge": 4,
    "ml.g5.24xlarge": 4,
    "ml.g5.48xlarge": 8,
    "ml.g4dn.xlarge": 1,
    "ml.g4dn.12xlarge": 4,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed XGBoost Fraud Detection on SageMaker")
    parser.add_argument("--role", type=str, required=True, help="SageMaker execution role ARN")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket")
    parser.add_argument("--region", type=str, default="us-west-2")
    parser.add_argument("--image-uri", type=str, default=None, help="XGBoost container image URI (default: auto-generated for region)")
    parser.add_argument("--instance-type", type=str, default="ml.g5.12xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--deploy-instance-type", type=str, default="ml.m5.large")
    parser.add_argument("--num-samples", type=int, default=500000, help="Number of synthetic transactions")
    parser.add_argument("--fraud-rate", type=float, default=0.02, help="Fraction of fraudulent transactions")
    parser.add_argument("--num-round", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--skip-cleanup", action="store_true")
    return parser.parse_args()


def generate_and_upload(bucket, region, num_samples, fraud_rate, instance_type, instance_count):
    """Generate synthetic fraud data, partition for Dask, upload to S3."""
    print(f"Generating {num_samples:,} synthetic transactions ({fraud_rate:.0%} fraud rate)...")
    X, y = make_classification(
        n_samples=num_samples,
        n_features=30,
        n_informative=15,
        n_redundant=5,
        weights=[1 - fraud_rate, fraud_rate],
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, "label", y)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=y, random_state=42)
    print(f"Train: {len(train_df):,}, Validation: {len(val_df):,}")
    print(f"Fraud count (train): {train_df['label'].sum():,} / {len(train_df):,}")

    # Partition for Dask: more files than total GPUs
    gpus = GPUS_PER_INSTANCE.get(instance_type, 1)
    total_gpus = gpus * instance_count
    num_partitions = max(total_gpus * 2, 8)
    print(f"Total GPUs: {total_gpus}, creating {num_partitions} partitions")

    s3_prefix = "xgboost-fraud-distributed"
    s3_client = boto3.client("s3", region_name=region)

    for split_name, split_df in [("train", train_df), ("validation", val_df)]:
        indices = np.array_split(np.arange(len(split_df)), num_partitions)
        for i, idx in enumerate(indices):
            part = split_df.iloc[idx]
            key = f"{s3_prefix}/{split_name}/part-{i:04d}.csv"
            csv_body = part.to_csv(index=False, header=False)
            s3_client.put_object(Bucket=bucket, Key=key, Body=csv_body)
        print(f"Uploaded {num_partitions} {split_name} partitions to s3://{bucket}/{s3_prefix}/{split_name}/")

    train_s3 = f"s3://{bucket}/{s3_prefix}/train/"
    val_s3 = f"s3://{bucket}/{s3_prefix}/validation/"
    scale_pos_weight = (1 - fraud_rate) / fraud_rate
    return train_s3, val_s3, scale_pos_weight


def train_model(role, bucket, region, instance_type, instance_count, train_s3, val_s3, scale_pos_weight, num_round, max_depth, image_uri=None):
    """Launch distributed GPU training with Dask."""
    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    xgb_image = image_uri or f"246618743249.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:3.0-5"
    print(f"XGBoost image: {xgb_image}")

    estimator = Estimator(
        image_uri=xgb_image,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        sagemaker_session=session,
        output_path=f"s3://{bucket}/xgboost-fraud-distributed/output",
        hyperparameters={
            "objective": "binary:logistic",
            "num_round": num_round,
            "max_depth": max_depth,
            "eta": 0.1,
            "tree_method": "gpu_hist",
            "scale_pos_weight": round(scale_pos_weight, 1),
            "eval_metric": "auc",
            "use_dask_gpu_training": "true",
        },
    )

    # FullyReplicated — Dask handles distribution internally
    train_input = TrainingInput(s3_data=train_s3, content_type="text/csv", distribution="FullyReplicated")
    val_input = TrainingInput(s3_data=val_s3, content_type="text/csv", distribution="FullyReplicated")

    estimator.fit({"train": train_input, "validation": val_input})
    print(f"Training complete. Model: {estimator.model_data}")
    return estimator


def deploy_and_test(estimator, deploy_instance_type):
    """Deploy to a CPU endpoint and run test inferences."""
    endpoint_name = f"xgb-fraud-{int(time.time())}"
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=deploy_instance_type,
        endpoint_name=endpoint_name,
        serializer=sagemaker.serializers.CSVSerializer(),
        deserializer=sagemaker.deserializers.CSVDeserializer(),
    )

    # Generate a few test samples
    rng = np.random.RandomState(99)
    test_features = rng.randn(5, 30)
    print("\n--- Sample Predictions ---")
    for i, row in enumerate(test_features):
        csv_line = ",".join(f"{v:.6f}" for v in row)
        result = predictor.predict(csv_line)
        score = float(result[0][0]) if isinstance(result[0], list) else float(result[0])
        label = "FRAUD" if score > 0.5 else "legit"
        print(f"  Transaction {i+1}: score={score:.4f} → {label}")

    return predictor, endpoint_name


def main():
    args = parse_args()

    print("=== Step 1: Generating and uploading data ===")
    train_s3, val_s3, scale_pos_weight = generate_and_upload(
        args.bucket, args.region, args.num_samples, args.fraud_rate, args.instance_type, args.instance_count
    )

    print(f"\n=== Step 2: Distributed GPU training ({args.instance_count}× {args.instance_type}) ===")
    estimator = train_model(
        args.role, args.bucket, args.region, args.instance_type, args.instance_count,
        train_s3, val_s3, scale_pos_weight, args.num_round, args.max_depth, args.image_uri,
    )

    if not args.skip_deploy:
        print("\n=== Step 3: Deploying and testing ===")
        predictor, endpoint_name = deploy_and_test(estimator, args.deploy_instance_type)

        if not args.skip_cleanup:
            print(f"\n=== Step 4: Cleaning up endpoint {endpoint_name} ===")
            predictor.delete_endpoint()
            print("Endpoint deleted.")
        else:
            print(f"\nEndpoint left running: {endpoint_name}")
    else:
        print("\nSkipped deployment. Model artifacts:", estimator.model_data)


if __name__ == "__main__":
    main()
