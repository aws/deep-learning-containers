import json
import logging
import os
import re
import sys
import time

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

        self.s3_client = boto3.client("s3")
        self.s3_resource = boto3.resource("s3")

        self.ticket_name_counter = 0
        self.request_lock = Lock()

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
            "SCHEDULING_TRIES": 0,
            "INSTANCES_NUM": num_of_instances,
            "TIMEOUT_LIMIT": self.timeout_limit,
            "COMMIT": os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", "default"),
        }

        return content

    def get_ticket_name_prefix(self):
        """
        Create a length 7 prefix for ticket name

        :return: <string> prefix for request ticket name
        """
        source_version = os.getenv("PR_NUMBER", "default")

        if "pr/" in source_version:
            # mod the PR ID by 100000 to make the prefix 7 digits
            return f"pr{(int(source_version.split('/')[-1]) % 100000):05}"
        else:
            return source_version[:7]

    def send_ticket(self, ticket_content, framework):
        """
        Send a request ticket to S3 bucket, self.s3_ticket_bucket

        Could run under multi-threading context, unique ticket name for each threads

        :param ticket_content: <dict> content of the ticket
        :return: <string> name of the ticket
        """
        LOGGER.info("send ticket is invoked")
        # ticket name: {CB source version}-{framework}{ticket name counter}_(datetime string)
        ticket_name_prefix = self.get_ticket_name_prefix()
        request_time = ticket_content["TIMESTAMP"]
        self.request_lock.acquire()
        ticket_name = f"{ticket_name_prefix}-{framework}{str(self.ticket_name_counter)}_{request_time}.json"
        self.ticket_name_counter += 1
        self.request_lock.release()
        self.s3_client.put_object(
            Bucket=self.s3_ticket_bucket,
            Key=f"{self.s3_ticket_bucket_folder}/{ticket_name}",
        )
        S3_ticket_object = self.s3_resource.Object(
            self.s3_ticket_bucket, f"{self.s3_ticket_bucket_folder}/{ticket_name}"
        )
        S3_ticket_object.put(Body=bytes(json.dumps(ticket_content).encode("UTF-8")))
        try:
            # change object acl to make ticket accessible to dev account.
            self.s3_client.put_object_acl(ACL="bucket-owner-full-control",Bucket=self.s3_ticket_bucket,
                Key=f"{self.s3_ticket_bucket_folder}/{ticket_name}")
        except Exception as e:
            raise e
        LOGGER.info(f"Ticket sent successfully, ticket name: {ticket_name}")
        return ticket_name

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

    def extract_timestamp(self, ticket_key):
        """
        extract the timestamp string from S3 request ticket key
        :param ticket_key: <string> key of the request ticket
        :return: <string> timestamp in format "%Y-%m-%d-%H-%M-%S" that is encoded in the ticket name
        """
        return re.match(r".*_(\d{4}(-\d{2}){5})\.json", ticket_key).group(1)

    def ticket_timestamp_cmp_function(self, ticket1_name, ticket2_name):
        """
        Compares the timestamp of the two request tickets

        :param ticket1, ticket2: <dict> S3 object descriptors from s3_client.list_objects
        :return: <bool>
        """
        ticket1_timestamp, ticket2_timestamp = (
            self.extract_timestamp(ticket1_name),
            self.extract_timestamp(ticket2_name),
        )
        return ticket1_timestamp > ticket2_timestamp

    def construct_query_response(self, status, reason=None, queueNum=None):
        """
        Create query response for query_status calls

        :param status: <string> queuing/preparing/completed/runtimeError
        :param reason: <string> maxRetries/timeout
        :param queueNum: <int>
        :return: <dict> response for the ticket query
        """
        query_response = {"status": status}
        if reason != None:
            query_response["reason"] = reason
        if queueNum != None:
            query_response["queueNum"] = queueNum

        return query_response

    def search_ticket_folder(self, folder, path):
        """
        Search folder/path on S3 to find the target ticket. If found, return a query response for the search. Otherwise
        return None.

        :param folder: <string> folder to search
        :param path: <string> path within the folder
        :return: <dict or None>
        """
        objects = self.s3_client.list_objects(Bucket=self.s3_ticket_bucket, Prefix=f"{folder}/{path}")
        if "Contents" in objects:
            ticket_key = objects["Contents"][0]["Key"]
            suffix_pattern = re.compile(".*-(.*).json")
            suffix = suffix_pattern.match(ticket_key).group(1)
            if folder == "dead_letter_queue" or folder == "duplicate_pr_requests":
                return self.construct_query_response("failed", reason=suffix)
            else:
                return self.construct_query_response(suffix)

        return None

    def send_request(self, image, build_context, num_of_instances):
        """
        Sending a request to test job executor (place request ticket to S3)

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
        framework = "mxnet" if "mxnet" in image else "pytorch" if "pytorch" in image else "tensorflow"
        ticket_name = self.send_ticket(ticket_content, framework)

        instance_type = self.assign_sagemaker_instance_type(image)
        job_type = "training" if "training" in image else "inference"
        identifier = Message(self.s3_ticket_bucket, ticket_name, image, instance_type, job_type, time)
        return identifier

    def receive_logs(self, identifier):
        """
        Requesting for the test logs

        :param identifier: <Message object> returned from send_request
        :return: <json or None> if log received, return the json log. Otherwise return None.
        """
        LOGGER.info("Receive logs called.")
        ticket_name_without_extension = identifier.ticket_name.rstrip(".json")
        objects = self.s3_client.list_objects(
            Bucket=self.s3_ticket_bucket,
            Prefix=f"resource_pool/{identifier.instance_type}-{identifier.job_type}/{ticket_name_without_extension}",
        )
        ticket_prefix = f"resource_pool/{identifier.instance_type}-{identifier.job_type}/{ticket_name_without_extension}"
        LOGGER.info(f"Receive Logs called, ticket prefix: {ticket_prefix}")
        if "Contents" in objects:
            entry = objects["Contents"][0]
            ticket_object = self.s3_client.get_object(Bucket="dlc-test-tickets", Key=entry["Key"])
            ticket_body = json.loads(ticket_object["Body"].read().decode("utf-8"))
            LOGGER.info("Ticket content successfully loaded.")
            return ticket_body["LOGS"]
g
        return None

    def cancel_request(self, identifier):
        """
        Cancel the test request by removing ticket from the queue.
        If the test request is already running, do nothing.

        :param identifier: <Message object> the response object returned from send_request
        """

        # check if ticket is on the queue
        ticket_in_queue = self.search_ticket_folder("request_tickets", identifier.ticket_name.rstrip(".json"))
        if ticket_in_queue:
            self.s3_client.delete_object(Bucket=self.s3_ticket_bucket, Key=f"request_tickets/{identifier.ticket_name}")
            return

        # check if ticket is a PR duplicate
        ticket_in_duplicate = self.search_ticket_folder("duplicate_pr_requests", identifier.ticket_name.rstrip(".json"))
        if ticket_in_duplicate:
            LOGGER.info(f"{identifier.ticket_name} is a duplicate PR test, test request will not be scheduled.")
            return

        LOGGER.info(f"{identifier.ticket_name} test has begun, test request could not be cancelled.")

    def query_status(self, identifier):
        """
        :param identifier: <Message object> unique identifier returned from call to send_request
        :return: <dict> {"status": <string> queuing/preparing/completed/failed/runtimeError,
                         "reason" (if status == failed): <string> maxRetries/timeout/duplicatePR,
                         "queueNum" (if status == queuing): <int>
                         }
        """
        retries = 2
        request_ticket_name = identifier.ticket_name
        ticket_without_extension = request_ticket_name.rstrip(".json")
        instance_type = identifier.instance_type
        job_type = identifier.job_type

        for _ in range(retries):
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
                    return self.construct_query_response("queuing", queueNum=queue_num)

            # check if ticket is on the dead letter queue
            ticket_in_dead_letter = self.search_ticket_folder("dead_letter_queue", ticket_without_extension)
            if ticket_in_dead_letter:
                return ticket_in_dead_letter

            ticket_in_duplicate = self.search_ticket_folder("duplicate_pr_requests", ticket_without_extension)
            if ticket_in_duplicate:
                return ticket_in_duplicate

            ticket_in_progress = self.search_ticket_folder(
                "resource_pool", f"{instance_type}-{job_type}/{ticket_without_extension}"
            )
            if ticket_in_progress:
                return ticket_in_progress

            time.sleep(2)

        raise AssertionError(f"Request ticket name {request_ticket_name} could not be found.")
