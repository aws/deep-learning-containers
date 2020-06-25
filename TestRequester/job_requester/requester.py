import json
import os
from datetime import datetime
from threading import Lock

import boto3

from job_requester.response import Message


class JobRequester():

    def __init__(self, timeout=14400):  # default timeout = 4 hours
        self.s3_ticket_bucket = "dlc-test-tickets"
        self.test_path = "test/sagemaker_tests/.../test_<feature>.py::test_function"
        self.timeout_limit = timeout

        self.sqs = boto3.client('sqs')
        self.S3 = boto3.client("s3")
        self.S3_resource = boto3.resource('s3')
        self.sqs_queue = self.create_SQS_queue()

        self.ticket_name_counter = 0
        self.logs = dict()
        self.l = Lock()

    def create_SQS_queue(self):
        """
		Create a SQS queue named with CODEBULD_BUILD_ID env variable

		:return: <string> SQS queue url
		"""
        # current build id as unique identifier for the SQS queue
        name = os.getenv("CODEBUILD_BUILD_ID").split(":")[1]
        response = self.sqs.create_queue(QueueName=name)
        queue_url = response["QueueUrl"]
        return queue_url

    def create_ticket_content(self, image, context, request_time):
        """
		Create the content of the ticket to be sent to S3

		:param image: ECR url
		:param context: build context
		:param request_time: <datetime string> time the request was created
		:return: <dict>
		"""
        content = {}
        content["CONTEXT"] = context
        content["TIMESTAMP"] = request_time
        content["TEST-PATH"] = self.test_path
        content["ECR-URI"] = image
        content["RETURN-SQS-URL"] = self.sqs_queue
        content["NUM_OF_SCHEDULING_TRIES"] = 0

        return content

    def send_ticket_to_S3(self, ticket_content, request_time):
        """
		Send a request ticket to S3 bucket, self.s3_ticket_bucket

		Could run under multithreading context, unique ticket name for each threads

		:param ticket_content:
		:return: <string> name of the ticket
		"""

        # create a unique ticekt name, CB execution ID - ticekt_name_counter _ datetime_str
        self.l.acquire()
        ticket_name = "{}-{}_{}.json".format(os.getenv("CODEBUILD_BUILD_ID").split(":")[1],
                                             str(self.ticket_name_counter), request_time)
        self.ticket_name_counter += 1
        self.l.release()
        # change to creating file locally,
        self.S3.put_object(Bucket=self.s3_ticket_bucket, Key=ticket_name)
        S3_ticket_object = self.S3_resource.Object(self.s3_ticket_bucket, ticket_name)
        S3_ticket_object.put(Body=bytes(json.dumps(ticket_content).encode('UTF-8')))

        return ticket_name

    def wait_for_SQS_message(self, ticket_name, request_time):
        """
		Polling SQS queue (self.queue_url) for messages, add received messages to self.logs

		Could run under multithreading context

		:param ticket_name: <string> name of the ticket for the test request
		:param request_time: <string> time that the request was placed
		:return:
		"""
        queue_response = self.sqs.receive_message(QueueUrl=self.sqs_queue)

        if "Messages" not in queue_response:
            # Do nothing
            return
        elif (datetime.strptime(request_time,
                                "%m/%d/%Y-%H:%M:%S") - datetime.now()).total_seconds() > self.timeout_limit:
            self.l.acquire()
            self.logs[ticket_name] = "Scheduling {} failed.".format(
                ticket_name)
            self.l.release()

        else:
            returned_messages = queue_response["Messages"]
            for message in returned_messages:
                self.l.acquire()
                log_message_in_json = json.loads(message["Body"])
                self.logs[log_message_in_json["TICKET_NAME"]] = log_message_in_json
                self.l.release()
                receipt_handle = message["ReceiptHandle"]
                try:
                    self.sqs.delete_message(QueueUrl=self.sqs_queue, ReceiptHandle=receipt_handle)
                except:
                    pass

    def send_request(self, image, build_context):
        """
		Sending a request to test job executor (set up SQS return queue and place ticket to S3)

		Could run under multithreading context

        :param image: <string> ECR uri
        :param build_context: <string> PR/MAINLINE/NIGHTLY/DEV
        :return: <Message object>
        """
        time = datetime.now().strftime("%m/%d/%Y-%H:%M:%S")
        ticket_content = self.create_ticket_content(image, build_context, time)
        ticket_name = self.send_ticket_to_S3(ticket_content, time)
        identifier = Message(self.sqs_queue, self.s3_ticket_bucket, ticket_name, image, time)
        return identifier


    def receive_logs(self, identifier):
        """
		Requesting for the test logs

        :param identifier: <Message object> returned from send_request
        :return:
        """
        ticket_name = identifier.ticket_name
        request_time = identifier.request_time

        if ticket_name in self.logs:
            return self.logs[ticket_name]

        else:
            res = self.wait_for_SQS_message(ticket_name, request_time)
            if ticket_name in self.logs:
                return self.logs[ticket_name]
            return "no returned log received."

    # TODO: this is only removing the tickets from S3. How to interrupt the test if it is already running?
    def cancel_request(self, identifier):
        """
		Cancel the test request

		:param identifier: the response object returned from send_request
		"""
        self.S3.delete_object(Bucket=self.s3_ticket_bucket, Key=identifier.ticket_name)

    def query_status (identifier):
        pass

    def clean_up(self):
        """
		Delete the SQS queue

		:return:
		"""
        self.sqs.delete_queue(QueueUrl=self.sqs_queue)
        return
