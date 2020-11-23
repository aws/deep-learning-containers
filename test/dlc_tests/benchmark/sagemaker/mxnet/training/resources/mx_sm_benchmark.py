import argparse
import os

import boto3
import sagemaker

from sagemaker.mxnet import MXNet


parser = argparse.ArgumentParser()
parser.add_argument("--framework-version", type=str, help="framework version in image to be used", required=True)
parser.add_argument("--image-uri", type=str, help="Image URI of image to benchmark", required=True)
parser.add_argument("--instance-type", type=str, help="instance type to use for test. Make sure to update processes_per_host according to the instance type.", required=True)
parser.add_argument("--node-count", type=int, help="number of nodes to train", default=4)
parser.add_argument("--python", help="python version", default="py3")
parser.add_argument("--region", help="region in which to run test", default="us-west-2")
parser.add_argument("--job-name", help="SageMaker Training Job Name", default=None)

args = parser.parse_args()

sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))

source_dir = "scripts"
processor = "gpu" if "gpu" in args.image_uri else "cpu"
entrypoint_script = "smtrain-resnet50-imagenet.sh"
processes_per_host = 8 if processor == "gpu" else 1
kwargs = {"train_volume_size": 200} if processor == "gpu" else {}

mx_estimator = MXNet(
    sagemaker_session=sagemaker_session,
    entry_point=entrypoint_script,
    source_dir=source_dir,
    role="SageMakerRole",
    instance_count=args.node_count,
    instance_type=args.instance_type,
    image_uri=args.image_uri,
    py_version=args.python,
    output_path=f"s3://dlc-bai-results-sagemaker-{args.region}",
    framework_version=args.framework_version,
    debugger_hook_config=False,
    distribution={
        "mpi": {
          "enabled": True,
          "processes_per_host": processes_per_host
        }
    },
    **kwargs
)

data = {
    # placeholder for data
    "s1": f"s3://dlc-data-sagemaker-{args.region}/small"
}

mx_estimator.fit(data, job_name=args.job_name, logs=True, wait=True)
