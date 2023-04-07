import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import datetime
import pandas as pd
import os
import argparse
import logging
import tarfile, subprocess

LOGGER = logging.getLogger(__name__)

DEFAULT_REGION = "us-west-2"

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str)
parser.add_argument("--job_name", type=str)
parser.add_argument("--instance_type", type=str)
parser.add_argument("--suites", type=str)

def get_boto3_session(region=DEFAULT_REGION):
    """Get boto3 session with us-east-1 as default region used to connect to AWS services."""
    return boto3.session.Session(region_name=region)

def get_cloudwatch_client(region=DEFAULT_REGION):
    """Get AWS CloudWatch client object. Currently assume region is IAD (us-east-1)"""
    return get_boto3_session(region=region).client("cloudwatch")

def put_metric_data(metric_name, namespace, unit, value, dimensions):
    """Puts data points to cloudwatch metrics"""
    cloudwatch_client = get_cloudwatch_client()
    current_timestamp = datetime.datetime.utcnow()
    try:
        response = cloudwatch_client.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {
                    "MetricName": metric_name,
                    "Dimensions": dimensions,
                    "Value": value,
                    "Unit": unit,
                    "Timestamp": current_timestamp,
                }
            ],
        )
    except ClientError as e:
        LOGGER.error("Error: Cannot put data to cloudwatch metric: {}".format(metric_name))
        LOGGER.error("Exception: {}".format(e))
        raise e

def read_metric(csv_file):
    df = pd.read_csv(csv_file)
    value = df[df.columns[-1]].iloc[0]
    if isinstance(value, str):
        for i in range(len(value)):
            if value[i].isdecimal() or value[i] == ".":
                continue
            else:
                return float(value[:i])
    return float(value)

if __name__ == '__main__':
    args = parser.parse_args()
    print("start uploading")
    dimensions = [
            {"Name": "InstanceType", "Value": args.instance_type},
            {"Name": "ModelSuite", "Value": args.suites},
            {"Name": "Precision", "Value": "AMP"},
            {"Name": "WorkLoad", "Value": "Training"},
    ]
    s3_artifact_path = os.path.join(args.output_path,args.job_name,"output","output.tar.gz")
    tmpdir = os.getcwd()
    local_artifact = os.path.join(tmpdir, "output.tar.gz")
    subprocess.check_output(["aws", "s3", "cp", s3_artifact_path, local_artifact])
    with tarfile.open(local_artifact, "r:gz") as result:
        result.extractall(path=tmpdir)
    result_path = os.path.join(tmpdir, "test","benchmark", "bin","pytorch",f"{args.suites}_logs")
    speedup = read_metric(os.path.join(result_path, "geomean.csv"))
    comp_time = read_metric(os.path.join(result_path,"comp_time.csv"))
    memory = read_metric(os.path.join(result_path,"memory.csv"))
    passrate = read_metric(os.path.join(result_path,"passrate.csv"))
    namespace = "PyTorch/SM/Benchmarks/TorchDynamo/Inductor"
    put_metric_data("Speedup", namespace, "None", speedup, dimensions)
    put_metric_data("CompilationTime", namespace, "Seconds", comp_time, dimensions)
    put_metric_data("PeakMemoryFootprintCompressionRatio", namespace, "None", memory, dimensions)
    put_metric_data("PassRate", namespace, "Percent", passrate, dimensions)
    print("finish uploading")