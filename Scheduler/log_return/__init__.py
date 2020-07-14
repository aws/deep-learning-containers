import boto3
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

def construct_log_content(report_path):
    """
    Create message that contains info allowing user to locate the logs

    :return: <json> returned message to SQS for locating the log
    """
    logs_client = boto3.client("logs")
    codebuild_arn = os.getenv("CODEBUILD_BUILD_ARN")
    log_group_name = "/aws/codebuild/DLCTestJobExecutor"
    log_stream_name = codebuild_arn.split(":")[-1]
    log_events = logs_client.get_log_events(logGroupName=log_group_name, logStreamName=log_stream_name)
    log_stream = "\n".join([event["message"] for event in log_events["events"]])

    try:
        with open(report_path) as xml_file:
            report_data = ET.parse(xml_file).getroot()
            report_data_in_string = ET.tostring(report_data).decode("utf-8")
    except FileNotFoundError as e:
        LOGGER.error(e)
        report_data_in_string = ""

    content = {
        "LOG_STREAM": log_stream,
        "XML_REPORT": report_data_in_string,
    }

    return content


def update_pool(status, instance_type, num_of_instances, job_type, report_path=None):
    """
    Update the S3 resource pool for usage of SageMaker resources.
    Naming convention of resource usage json: request ticket_name#num_of_instances-status.

    :param job_type: <string> training/inference
    :param report_path: <string> path to find the xml reports. Only set if status == completed/runtimeError
    :param status: status of the test job, options: preparing/running/completed/runtimeError
    :param instance_type: ml.p3.8xlarge/ml.c4.4xlarge/ml.p2.8xlarge/ml.c4.8xlarge
    :param num_of_instances: number of instances required
    """
    s3_client = boto3.client("s3")
    codebuild_arn = os.getenv("CODEBUILD_BUILD_ARN")
    ticket_name = os.getenv("TICKET_KEY").split("/")[-1].split(".")[0]

    if status not in {"preparing", "running", "completed", "runtimeError"}:
        raise ValueError("Not a valid status. Test job status could be preparing, running, completed or runtimeError.")

    pool_ticket_content = {
        "REQUEST_TICKET_KEY": os.getenv("TICKET_KEY"),
        "STATUS": status,
        "INSTANCE_TYPE": instance_type,
        "EXECUTOR_ARN": codebuild_arn,
        "INSTANCES_NUM": num_of_instances,
    }

    if status == "completed" or status == "runtimeError":
        pool_ticket_content["LOGS"] = construct_log_content(report_path)

    # find previous entry of the test job
    response = s3_client.list_objects(
        Bucket="dlc-test-tickets", MaxKeys=1, Prefix=f"resource_pool/{instance_type}-{job_type}/{ticket_name}"
    )

    # creating json file locally and upload to S3
    filename = f"{ticket_name}#{num_of_instances}-{status}.json"
    with open(filename, "w") as f:
        json.dump(pool_ticket_content, f)

    with open(filename, "rb") as data:
        s3_client.upload_fileobj(data, "dlc-test-tickets", f"resource_pool/{instance_type}-{job_type}/{filename}")

    # delete previous entry of the test job. Note: the deletion is performed after uploading a new ticket to avoid
    # S3's Eventual Consistency causes any issues with finding the state of a ticket during a state-transition
    if "Contents" in response:
        previous_entry = response["Contents"][0]
        s3_client.delete_object(Bucket="dlc-test-tickets", Key=previous_entry["Key"])
