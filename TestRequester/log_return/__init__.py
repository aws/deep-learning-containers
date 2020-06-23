import os
import boto3
import xmltodict
import json


def log_locater(report_path):
    """
    Create message that contains info allowing user to locate the logs

    :return: <json> returned message to SQS for locating the log
    """
    arn = os.getenv("CODEBUILD_BUILD_ARN")
    log_group_name = "/aws/codebuild/" + arn.split(":")[-2]
    log_stream_name = arn.split(":")[-1]

    with open(report_path) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
        xml_file.close()
        report_json_data = json.dumps(data_dict)

    content = {}
    content["LOG_GROUP_NAME"] = log_group_name
    content["LOG_STREAM_NAME"] = log_stream_name
    content["TICKET_NAME"] = os.getenv("TICKET_NAME")
    content["XML_REPORT"] = report_json_data

    return json.dumps(content)


def send_log(report_path):
    """
    Sending log message to SQS
    """
    print(os.getcwd())
    log_sqs_url = os.getenv("RETURN_SQS_URL")
    log_location = log_locater(report_path)
    sqs_client = boto3.client("sqs")
    sqs_client.send_message(QueueUrl=log_sqs_url, MessageBody=log_location)
    print(f"Logs successfully sent to {log_sqs_url}")
