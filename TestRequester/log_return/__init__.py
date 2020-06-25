import os
import boto3
import xmltodict
import json


def log_locater(report_path):
    """
    Create message that contains info allowing user to locate the logs

    :return: <json> returned message to SQS for locating the log
    """
    codebuild_arn = os.getenv("CODEBUILD_BUILD_ARN")
    ticket_name = os.getenv("TICKET_NAME").split("/")[1].split(".")[0]
    log_group_name = "/aws/codebuild/" + codebuild_arn.split(":")[-2]
    log_stream_name = codebuild_arn.split(":")[-1]

    with open(report_path) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
        xml_file.close()
        report_json_data = json.dumps(data_dict)

    content = {}
    content["LOG_GROUP_NAME"] = log_group_name
    content["LOG_STREAM_NAME"] = log_stream_name
    content["TICKET_NAME"] = ticket_name
    content["XML_REPORT"] = report_json_data

    return json.dumps(content)


def send_log(report_path):
    """
    Sending log message to SQS
    """
    log_sqs_url = os.getenv("RETURN_SQS_URL")
    sqs_client = boto3.client("sqs")
    log_location = log_locater(report_path)
    sqs_client.send_message(QueueUrl=log_sqs_url, MessageBody=log_location)
    print(f"Logs successfully sent to {log_sqs_url}")


def update_pool(status, instance_type, num_of_instances=1):
    """
    Update the S3 resource pool for usage of SageMaker resources.
    Naming convention of resource usage json: ticket_name-status.

    :param status: status of the test job, options: preparing/running/completed/failed
    :param instance_type: ml.p3.8xlarge/ml.c4.4xlarge/ml.p2.8xlarge/ml.c4.8xlarge
    :param num_of_instances: number of instances required
    """
    s3_client = boto3.client("s3")
    codebuild_arn = os.getenv("CODEBUILD_BUILD_ARN")
    ticket_name = os.getenv("TICKET_NAME").split("/")[1].split(".")[0]

    if status not in {"preparing", "running", "completed", "failed"}:
        raise ValueError("Not a valid status. Test job status could be preparing, running, completed or failed.")

    pool_ticket_content = {}
    pool_ticket_content["TICKET_NAME"] = ticket_name
    pool_ticket_content["STATUS"] = status
    pool_ticket_content["INSTANCE_TYPE"] = instance_type
    pool_ticket_content["EXECUTOR_ARN"] = codebuild_arn
    pool_ticket_content["INSTANCES_NUM"] = num_of_instances

    # delete existing entries of the job, if present
    response = s3_client.list_objects(Bucket="dlc-test-tickets", MaxKeys=1,
                                      Prefix=f"resource_pool/{instance_type}/{ticket_name}")
    if "Contents" in response:
        previous_entry = response["Contents"][0]
        s3_client.delete_object(Bucket="dlc-test-tickets", Key=previous_entry["Key"])

    # creating json file locally and upload to S3
    filename = f"{ticket_name}-{status}.json"
    with open(filename, "w") as f:
        json.dump(pool_ticket_content, f)

    with open(filename, "rb") as data:
        s3_client.upload_fileobj(data, "dlc-test-tickets", f"resource_pool/{instance_type}/{filename}")
