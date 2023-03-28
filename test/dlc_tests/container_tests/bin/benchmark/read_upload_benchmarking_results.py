import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import datetime
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-dir", "--directory", type=str)


def get_boto3_session(region="us-east-1"):
    """Get boto3 session with us-east-1 as default region used to connect to AWS services."""
    return boto3.session.Session(region_name=region)

def get_cloudwatch_client(region="us-east-1"):
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

def upload_metric(unique_cw_namespace, value, unique_metric_dims):
    def _upload_metric(metric_name, value, unit):
        put_metric_data(
                 metric_name=metric_name,
                 namespace=unique_cw_namespace,
                 unit=unit,
                 value=value,
                 dimensions=unique_metric_dims,
        )

    return _upload_metric

if __name__ == '__main__':
    args = parser.parse_args()
    speedup = read_metric(os.path.join(args.directory, "geomean.csv"))
    comp_time = read_metric(os.path.join(args.directory,"comp_time.csv"))
    memory = read_metric(os.path.join(args.directory,"memory.csv"))
    passrate = read_metric(os.path.join(args.directory,"passrate.csv"))
    upload_metric("Speedup", speedup, "None")
    upload_metric("CompilationTime", comp_time, "Seconds")
    upload_metric("PeakMemoryFootprintCompressionRatio", memory, "None")
    upload_metric("PassRate", passrate, "Percent")