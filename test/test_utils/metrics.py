import os

from datetime import datetime

import boto3


def construct_duration_metrics_data(start_time, test_path):
    duration = (datetime.now() - start_time).total_seconds()
    data = {
        "MetricName": "Test Duration",
        "Dimensions": [{"Name": "Test Path", "Value": test_path}],
        "Value": duration,
    }
    return data


def construct_test_result_metrics_data(stdout, test_path):
    data = {
        "MetricName": "Test Errors",
        "Dimensions": [{"Name": "Test Path", "Value": test_path}],
        "Value": stdout,
    }
    return data


def send_test_duration_metrics(start_time):
    cloudwatch_client = boto3.client("cloudwatch")
    use_scheduler = os.getenv("USE_SCHEDULER", "False")
    executor_mode = os.getenv("EXECUTOR_MODE", "False")

    if executor_mode.lower() == "false":
        if use_scheduler.lower() == "true":
            metric_data = construct_duration_metrics_data(start_time, "With Scheduler")

        elif use_scheduler.lower() == "false":
            metric_data = construct_duration_metrics_data(start_time, "Without Scheduler")

        cloudwatch_client.put_metric_data(Namespace="DLCCI", MetricData=[metric_data])


def send_test_result_metrics(stdout):
    cloudwatch_client = boto3.client("cloudwatch")

    use_scheduler = os.getenv("USE_SCHEDULER", "False")
    executor_mode = os.getenv("EXECUTOR_MODE", "False")
    if executor_mode.lower() == "false":
        if use_scheduler.lower() == "true":
            metric_data = construct_test_result_metrics_data(stdout, "With Scheduler")

        elif use_scheduler.lower() == "false":
            metric_data = construct_test_result_metrics_data(stdout, "Without Scheduler")

        cloudwatch_client.put_metric_data(Namespace="DLCCI", MetricData=[metric_data])

