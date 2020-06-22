import os
import boto3



def log_locater():
    arn = os.getenv("CODEBUILD_BUILD_ARN")
    log_group_name = "/aws/codebuild/" + arn.split(":")[-2]
    log_stream_name = arn.split(":")[-1]

    with open("test/sagemaker.xml") as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
        xml_file.close()
        report_json_data = json.dumps(data_dict)

    content = {}
    content["LOG_GROUP_NAME"] = log_group_name
    content["LOG_STREAM_NAME"] = log_stream_name
    content["TICKET_NAME"] = os.getenv("TICKET_NAME")
    content["XML_REPORT"] = report_json_data

    return json.dumps(content)

def send_log():
    log_sqs_url = os.getenv("RETURN_SQS_URL")
    log_location = log_locater()
    sqs = boto3.client("sqs")
    print(log_sqs_url)
    sqs.send_message(QueueUrl=log_sqs_url, MessageBody=log_location)
