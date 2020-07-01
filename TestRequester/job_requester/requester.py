import json
import logging
import os
import re
import sys

from datetime import datetime
from threading import Lock
from functools import cmp_to_key

import boto3

from job_requester import Message

MAX_TIMEOUT_IN_SEC = 14400  # 4 hours

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


class JobRequester:
    def __init__(self, timeout=MAX_TIMEOUT_IN_SEC):
        self.s3_ticket_bucket = "dlc-test-tickets"
        self.s3_ticket_bucket_folder = "request_tickets"
        self.timeout_limit = min(timeout, MAX_TIMEOUT_IN_SEC)

        self.sqs_client = boto3.client("sqs")
        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3")
        self.sqs_queue = self.create_sqs_queue()

        self.ticket_name_counter = 0
        self.logs = dict()
        self.request_lock = Lock()

    def __del__(self):
        # clean up the SQS return queue
        self.sqs_client.delete_queue(QueueUrl=self.sqs_queue)

    def create_sqs_queue(self):
        """
        Create a SQS queue named with CODEBUILD_BUILD_ID env variable

        :return: <string> SQS queue url
        """
        # current build id as unique identifier for the SQS queue
        name = os.getenv("CODEBUILD_BUILD_ID", "TestRequester:default").split(":")[-1]
        response = self.sqs_client.create_queue(QueueName=name)
        queue_url = response["QueueUrl"]
        return queue_url

    def create_ticket_content(self, image, context, num_of_instances, request_time):
        """
        Create content of the ticket to be sent to S3

        :param image: <string> ECR URI
        :param context: <string> build context (PR/MAINLINE/NIGHTLY/DEV)
        :param num_of_instances: <int> number of instances required by the test job
        :param request_time: <string> datetime timestamp of when request was made
        :return: <dict> content of the request ticket
        """
        content = {
            "CONTEXT": context,
            "TIMESTAMP": request_time,
            "ECR-URI": image,
            "RETURN-SQS-URL": self.sqs_queue,
            "SCHEDULING_TRIES": 0,
            "INSTANCES_NUM": num_of_instances,
            "TIMEOUT_LIMIT": self.timeout_limit,
        }

        return content

    def get_ticket_name_prefix(self):
        """
        Create a length 7 prefix for ticket name

        :return: <string> prefix for request ticket name
        """
        source_version = os.getenv("CODEBUILD_SOURCE_VERSION", "default")

        if "pr/" in source_version:
            # mod the PR ID by 10000 to make the prefix 7 digits
            return f"pr{(int(source_version.split('/')[-1]) % 100000):05}"
        else:
            return source_version[:7]

    def send_ticket(self, ticket_content):
        """
        Send a request ticket to S3 bucket, self.s3_ticket_bucket

        Could run under multi-threading context, unique ticket name for each threads

        :param ticket_content: <dict> content of the ticket
        :return: <string> name of the ticket
        """

        # ticket name: {CB source version}-{ticket name counter}_(datetime string)
        ticket_name_prefix = self.get_ticket_name_prefix()
        request_time = ticket_content["TIMESTAMP"]
        self.request_lock.acquire()
        ticket_name = f"{ticket_name_prefix}-{str(self.ticket_name_counter)}_{request_time}.json"
        self.ticket_name_counter += 1
        self.request_lock.release()

        self.s3_client.put_object(Bucket=self.s3_ticket_bucket, Key=f"{self.s3_ticket_bucket_folder}/{ticket_name}")
        S3_ticket_object = self.s3_resource.Object(
            self.s3_ticket_bucket, f"{self.s3_ticket_bucket_folder}/{ticket_name}"
        )
        S3_ticket_object.put(Body=bytes(json.dumps(ticket_content).encode("UTF-8")))

        return ticket_name

    def receive_sqs_message(self):
        """
        Polling SQS queue (self.queue_url) for messages, add received messages to self.logs
        Could run under multi-threading context

        :return: None
        """
        queue_response = self.sqs_client.receive_message(QueueUrl=self.sqs_queue)

        if "Messages" not in queue_response:
            # Do nothing
            return

        else:
            returned_messages = queue_response["Messages"]
            for message in returned_messages:
                self.request_lock.acquire()
                log_message_in_json = json.loads(message["Body"])
                self.logs[log_message_in_json["TICKET_NAME"]] = log_message_in_json
                self.request_lock.release()
                receipt_handle = message["ReceiptHandle"]
                try:
                    self.sqs_client.delete_message(QueueUrl=self.sqs_queue, ReceiptHandle=receipt_handle)
                except self.sqs_client.exceptions.ReceiptHandleIsInvalid as e:
                    LOGGER.warning(f"Not the latest ReceiptHandle, message could already been deleted: {e}")

    def send_request(self, image, build_context, num_of_instances):
        """
        Sending a request to test job executor (set up SQS return queue and place ticket to S3)

        Could run under multi-threading context

        :param num_of_instances: <int> number of instances needed for the test
        :param image: <string> ECR uri
        :param build_context: <string> PR/MAINLINE/NIGHTLY/DEV
        :return: <Message object>
        """
        assert (
            "training" in image or "inference" in image
        ), f"Job type (training/inference) not stated in image tag: {image}"
        time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        ticket_content = self.create_ticket_content(image, build_context, num_of_instances, time)
        ticket_name = self.send_ticket(ticket_content)
        identifier = Message(self.sqs_queue, self.s3_ticket_bucket, ticket_name, image, time)
        return identifier

    def receive_logs(self, identifier):
        """
        Requesting for the test logs

        :param identifier: <Message object> returned from send_request
        :return: <json or None> if log received, return the json log. Otherwise return None.
        """
        ticket_name = identifier.ticket_name

        if ticket_name in self.logs:
            return self.logs[ticket_name]

        else:
            self.receive_sqs_message()
            if ticket_name in self.logs:
                return self.logs[ticket_name]
            return None

    def cancel_request(self, identifier):
        """
        Cancel the test request by removing ticket from the queue.
        If the test request is already running, do nothing.

        :param identifier: <Message object> the response object returned from send_request
        """
        ticket_objects = self.s3_client.list_objects(
            Bucket=self.s3_ticket_bucket, Prefix=f"request_tickets/{identifier.ticket_name}"
        )
        if "Contents" in ticket_objects:
            self.s3_client.delete_object(Bucket=self.s3_ticket_bucket, Key=f"request_tickets/{identifier.ticket_name}")
            return
        LOGGER.info(f"{identifier.ticket_name} is already running, test request could not be cancelled.")

    def assign_sagemaker_instance_type(self, image):
        """
        Assign the instance type that the input image needs for testing

        :param image: <string> ECR URI
        :return: <string> type of instance used by the image
        """
        if "tensorflow" in image:
            return "ml.p3.8xlarge" if "gpu" in image else "ml.c4.4xlarge"
        else:
            return "ml.p2.8xlarge" if "gpu" in image else "ml.c4.8xlarge"

    def ticket_timestamp_cmp_function(self, ticket1, ticket2):
        """
        Compares the timestamp of the two request tickets

        :param ticket1, ticket2: <dict> S3 object descriptors from s3_client.list_objects
        :return: <bool>
        """
        timestamp_pattern = re.compile(".*_(.*).json")
        ticket1_time, ticket2_time = (
            timestamp_pattern.match(ticket1).group(1),
            timestamp_pattern.match(ticket2).group(1),
        )
        return ticket1_time > ticket2_time

    def create_query_response(self, status, reason=None, queueNum=None):
        """
        Create query response for query_status calls
        
        :param status: <string> queuing/preparing/completed/failed
        :param reason: <string> maxRetries/timeout
        :param queueNum: <int>
        """
        query_response = {"status": status}
        if reason != None:
            query_response["reason"] = reason
        if queueNum != None:
            query_response["queueNum"] = queueNum

        return query_response

    def query_status(self, identifier):
        """
        :param identifier: <Message object> unique identifier returned from call to send_request
        :return: <dict> {"status": <string> queuing/preparing/completed/failed,
                         "reason" (if status == failed): <string> maxRetries/timeout,
                         "queueNum" (if status == queuing): <int>
                         }
        """
        request_ticket_name = identifier.ticket_name
        request_image = identifier.image
        instance_type = self.assign_sagemaker_instance_type(request_image)
        job_type = "training" if "training" in request_image else "inference"

        # check if ticket is on the queue
        ticket_objects = self.s3_client.list_objects(Bucket=self.s3_ticket_bucket, Prefix="request_tickets/")
        # "Contents" in the API response only if there are objects satisfy the prefix
        if "Contents" in ticket_objects:
            ticket_name_pattern = re.compile(".*\/(.*)")
            ticket_names_list = [
                ticket_name_pattern.match(ticket["Key"]).group(1)
                for ticket in ticket_objects["Contents"]
                if ticket["Key"].endswith(".json")
            ]
            # ticket is on the queue, find the queue number
            if request_ticket_name in ticket_names_list:
                ticket_names_list.sort(key=cmp_to_key(self.ticket_timestamp_cmp_function))
                queue_num = ticket_names_list.index(request_ticket_name)
                return self.create_query_response("queuing", queueNum=queue_num)

        # check if ticket is on the dead letter queue
        dead_letter_objects = self.s3_client.list_objects(
            Bucket=self.s3_ticket_bucket, Prefix=f"dead_letter_queue/{request_ticket_name}"
        )
        if "Contents" in dead_letter_objects:
            dead_letter_ticket_key = dead_letter_objects["Contents"][0]["Key"]
            failed_reason_pattern = re.compile(".*-(.*).json")
            failed_reason = failed_reason_pattern.match(dead_letter_ticket_key).group(1)
            return self.create_query_response("failed", reason=failed_reason)

        # check if ticket is on the in-progress pool
        in_progress_objects = self.s3_client.list_objects(
            Bucket=self.s3_ticket_bucket, Prefix=f"resource_pool/{instance_type}-{job_type}/{request_ticket_name}"
        )
        if "Contents" in in_progress_objects:
            in_progress_ticket_key = in_progress_objects["Contents"][0]["Key"]
            in_progress_status_pattern = re.compile(".*-(.*).json")
            in_progress_status = in_progress_status_pattern.match(in_progress_ticket_key).group(1)
            return self.create_query_response(in_progress_status)

        raise AssertionError(f"Request ticket name {request_ticket_name} could not be found.")
