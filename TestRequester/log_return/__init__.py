import os
import boto3
import xmltodict
import json

def LogReturn():

    def __init__(self):
        self.sqs_client = boto3.client("sqs")
        self.s3_client = boto3.client("s3")

        self.ticket_name = os.getenv("TICKET_NAME")
        self.log_sqs_url = os.getenv("RETURN_SQS_URL")
        self.codebuild_arn = os.getenv("CODEBUILD_BUILD_ARN")

    def log_locater(self, report_path):
        """
        Create message that contains info allowing user to locate the logs

        :return: <json> returned message to SQS for locating the log
        """
        log_group_name = "/aws/codebuild/" + self.codebuild_arn.split(":")[-2]
        log_stream_name = self.codebuild_arn.split(":")[-1]

        with open(report_path) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
            xml_file.close()
            report_json_data = json.dumps(data_dict)

        content = {}
        content["LOG_GROUP_NAME"] = log_group_name
        content["LOG_STREAM_NAME"] = log_stream_name
        content["TICKET_NAME"] = self.ticket_name
        content["XML_REPORT"] = report_json_data

        return json.dumps(content)


    def send_log(self, report_path):
        """
        Sending log message to SQS
        """
        log_location = self.log_locater(report_path)
        self.sqs_client.send_message(QueueUrl=self.log_sqs_url, MessageBody=log_location)
        print(f"Logs successfully sent to {self.log_sqs_url}")


    def update_pool(self, status, instance_type, num_of_instances):
        pool_ticket_content = {}
        pool_ticket_content["TICKET_NAME"] = self.ticket_name
        pool_ticket_content["STATUS"] = status  #preparing/running/success/failed
        pool_ticket_content["INSTANCE_TYPE"] = instance_type
        pool_ticket_content["EXECUTOR_ARN"] = self.codebuild_arn
        pool_ticket_content["INSTANCES_NUM"] = num_of_instances

        #delete existing entries of the job, if present
        previous_entry = self.s3_client.list_objects(Bucket="dlc-test-tickets", MaxKeys=1, Prefix=f"resource_pool/{instance_type}/{self.ticket_name}")["Contents"]
        if len(previous_entry) != 0:
            self.s3_client.delete_object(Bucket="dlc-test-tickets", Key=previous_entry[0]["Key"])

        #creating json file and upload to S3
        filename = f"{self.ticket_name}-{status}"
        with open(filename, "w") as f:
            json.dump(pool_ticket_content, f)
            self.s3_client.upload_fileobj(f, "dlc-test-tickets", f"resource_pool/{instance_type}/{filename}")









