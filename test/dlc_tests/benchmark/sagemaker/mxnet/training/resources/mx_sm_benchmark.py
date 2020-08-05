import argparse
import os

import boto3
import sagemaker

from sagemaker.mxnet import MXNet


parser = argparse.ArgumentParser()
parser.add_argument("--framework-version", type=str, help="framework version in image to be used", required=True)
parser.add_argument("--image-uri", type=str, help="Image URI of image to benchmark", required=True)
parser.add_argument("--instance-type", type=str, help="instance type to use for test", required=True)
parser.add_argument("--node-count", type=int, help="number of nodes to train", default=4)
parser.add_argument("--python", help="python version", default="py3")
parser.add_argument("--region", help="region in which to run test", default="us-west-2")
parser.add_argument("--job-name", help="SageMaker Training Job Name", default=None)

args = parser.parse_args()

sagemaker_session = sagemaker.Session(boto3.Session(region_name=args.region))

source_path = "scripts"

mx_estimator = MXNet(
    sagemaker_session=sagemaker_session,
    entry_point="smtrain-resnet50-imagenet.sh",
    source_dir=source_path,
    role="SageMakerRole",
    train_instance_count=args.node_count,
    train_instance_type=args.instance_type,
    image_name=args.image_uri,
    py_version=args.python,
    output_path=f"s3://bapac-chexpert-mini/",
    train_volume_size=200,
    framework_version=args.framework_version,
    distributions={"mpi": {"enabled": True}},
)

data = {
    # placeholder for data
    "s1": f"s3://dlc-data-sagemaker-{args.region}/small"
}

mx_estimator.fit(data, job_name=args.job_name, logs=True, wait=True)