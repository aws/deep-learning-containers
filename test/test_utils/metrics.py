import os

from datetime import datetime

import boto3


def construct_duration_metrics_data(start_time, test_path):
    """
    construct test duration metrics data to be sent to cloudwatch
    :param start_time: <datetime> start time of the test execution
    :param test_path: <string> With Scheduler/Without Scheduler
    :return: <dict>
    """
    duration = (datetime.now() - start_time).total_seconds()
    data = {"MetricName": "Test Duration", "Dimensions": [{"Name": "Test Path", "Value": test_path}], "Value": duration}
    return data


def construct_test_result_metrics_data(stdout, test_path):
    """
    construct test results metrics data to be sent to cloudwatch
    :param stdout: <int> 0/1. 0 indicates no error during test execution, 1 indicates errors occurred
    :param test_path: <string> With Scheduler/Without Scheduler
    :return: <dict>
    """
    data = {"MetricName": "Test Errors", "Dimensions": [{"Name": "Test Path", "Value": test_path}], "Value": stdout}
    return data


def send_test_duration_metrics(start_time):
    """
    send custom metrics about test duration to cloudwatch
    :param start_time: <datetime> start time of the test execution
    """
    cloudwatch_client = boto3.client("cloudwatch")
    use_scheduler = os.getenv("USE_SCHEDULER", "False").lower() == "true"
    executor_mode = os.getenv("EXECUTOR_MODE", "False").lower() == "true"
    if not executor_mode:  # metrics should only be sent by the test CB
        if use_scheduler:
            metric_data = construct_duration_metrics_data(start_time, "With Scheduler")

        else:
            metric_data = construct_duration_metrics_data(start_time, "Without Scheduler")

        cloudwatch_client.put_metric_data(Namespace="DLCCI", MetricData=[metric_data])


def send_test_result_metrics(stdout):
    """
    Send custom metrics about test results to cloudwatch.
    :param stdout: <int> 0/1. 0 indicates no error during test execution, 1 indicates errors occurred
    """
    cloudwatch_client = boto3.client("cloudwatch")
    use_scheduler = os.getenv("USE_SCHEDULER", "False").lower() == "true"
    executor_mode = os.getenv("EXECUTOR_MODE", "False").lower() == "true"
    if not executor_mode: # metrics should only be sent by the test CB
        if use_scheduler:
            metric_data = construct_test_result_metrics_data(stdout, "With Scheduler")

        else:
            metric_data = construct_test_result_metrics_data(stdout, "Without Scheduler")

        cloudwatch_client.put_metric_data(Namespace="DLCCI", MetricData=[metric_data])
