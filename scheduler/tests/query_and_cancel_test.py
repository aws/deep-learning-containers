import json
import logging
import sys

from datetime import datetime

import boto3

from job_requester import JobRequester
from job_requester import Message


"""
How tests are executed:
- Put tickets on the request queue, in-progress pool and dead letter queue; create Message objects (request identifiers)
that correspond to these tickets.
- Create a JobRequester object and call query_status on the identifiers, check that the statuses returned are corrected.
- Call cancel_request on tickets on the request queue, check that the request tickets are removed. 
- Clean up the artifacts. 
"""

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

# test parameters
TEST_ECR_URI = "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.2.0-gpu-py37-cu101-ubuntu18.04"
INSTANCE_TYPE = "ml.p3.8xlarge"
JOB_TYPE = "training"
SQS_RETURN_QUEUE_URL = "dummy_sqs_url"
REQUEST_TICKET_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
REQUEST_TICKET_CONTENT = {
    "CONTEXT": "PR",
    "TIMESTAMP": REQUEST_TICKET_TIME,
    "ECR-URI": TEST_ECR_URI,
    "RETURN-SQS-URL": SQS_RETURN_QUEUE_URL,
    "SCHEDULING_TRIES": 0,
    "INSTANCES_NUM": 1,
    "TIMEOUT_LIMIT": 14400,
}
IN_PROGRESS_TICKET_CONTENT = {
    "INSTANCE_TYPE": INSTANCE_TYPE,
    "INSTANCES_NUM": 1,
    "STATUS": "preparing",
}
DEAD_LETTER_TICKET_CONTENT = {
    "INSTANCE_TYPE": INSTANCE_TYPE,
    "INSTANCES_NUM": 1,
    "STATUS": "failed",
}

# S3 path to tickets
BUCKET_NAME = "dlc-test-tickets"
REQUEST_TICKETS_FOLDER = "request_tickets"
IN_PROGRESS_POOL_FOLDER = "resource_pool"
DEAD_LETTER_QUEUE_FOLDER = "dead_letter_queue"


def put_ticket(ticket_key, ticket_content):
    """
    API calls to put and write to a ticket on S3
    :param ticket_key: <string> S3 path to the ticket
    :param ticket_content: <dict>
    """
    s3_client = boto3.client("s3")
    s3_resource = boto3.resource("s3")
    s3_client.put_object(Bucket=BUCKET_NAME, Key=ticket_key)
    S3_ticket_object = s3_resource.Object(BUCKET_NAME, ticket_key)
    S3_ticket_object.put(Body=bytes(json.dumps(ticket_content).encode("UTF-8")))


def clean_up_ticket(ticket_key):
    """
    Clean up testing ticket artifacts
    :param ticket_key: <string> S3 path to ticket
    """
    s3_client = boto3.client("s3")
    s3_client.delete_object(Bucket=BUCKET_NAME, Key=ticket_key)


def test_query_and_cancel_queuing_tickets(job_requester, request_queue_ticket_name, request_identifier):
    """
    test querying and cancelling tickets that are yet to be scheduled
    :param job_requester: JobRequester object
    :param request_queue_ticket_name: request ticket name
    :param request_identifier: <Message object> identifier for the request sent
    """
    s3_client = boto3.client("s3")
    put_ticket(f"{REQUEST_TICKETS_FOLDER}/{request_queue_ticket_name}", REQUEST_TICKET_CONTENT)
    # check the response message is correct
    request_queue_response = job_requester.query_status(request_identifier)
    assert (
        request_queue_response["status"] == "queuing"
    ), f"Returned status incorrect: {request_queue_ticket_name}, should be queuing."
    assert (
        "queueNum" in request_queue_response
    ), f"Queue number not found for request ticket {request_queue_ticket_name}"

    # test cancelling request for tickets on the queue
    job_requester.cancel_request(request_identifier)
    # check that the request ticket has been removed
    list_request_ticket_response = s3_client.list_objects(
        Bucket=BUCKET_NAME, Prefix=f"request_tickets/{request_queue_ticket_name}"
    )
    # if there is no object that satisfies list_objects conditions, "Contents" field would not be in the API response
    assert (
        "Contents" not in list_request_ticket_response
    ), f"Request ticket {request_queue_ticket_name} not correctly cancelled."

    LOGGER.info("Tests passed for querying and cancelling tickets on the queue.")


def test_query_in_progress_tickets(job_requester, in_progress_ticket_name, request_identifier):
    """
    test querying test jobs that are scheduled and running
    :param job_requester: JobRequester object
    :param request_queue_ticket_name: request ticket name
    :param request_identifier: <Message object> identifier for the request sent
    """
    put_ticket(
        f"{IN_PROGRESS_POOL_FOLDER}/{INSTANCE_TYPE}-{JOB_TYPE}/{in_progress_ticket_name}", IN_PROGRESS_TICKET_CONTENT
    )

    in_progress_response = job_requester.query_status(request_identifier)
    assert (
        "status" in in_progress_response and in_progress_response["status"] == "running"
    ), f"Returned status incorrect: {in_progress_ticket_name}, should be running."

    clean_up_ticket(f"{IN_PROGRESS_POOL_FOLDER}/{INSTANCE_TYPE}-{JOB_TYPE}/{job_requester}")

    LOGGER.info("Tests passed for querying tickets on the in-progress pool.")


def test_query_dead_letter_tickets(job_requester, dead_letter_ticket_name, request_identifier):
    """
    test querying test jobs that are failed to be scheduled
    :param job_requester: JobRequester object
    :param request_queue_ticket_name: request ticket name
    :param request_identifier: <Message object> identifier for the request sent
    """
    put_ticket(f"{DEAD_LETTER_QUEUE_FOLDER}/{dead_letter_ticket_name}", DEAD_LETTER_TICKET_CONTENT)

    dead_letter_response = job_requester.query_status(request_identifier)
    assert (
        "status" in dead_letter_response and dead_letter_response["status"] == "failed"
    ), f"Returned status incorrect: {dead_letter_ticket_name}, should be failed."
    assert (
        "reason" in dead_letter_response and dead_letter_response["reason"] == "timeout"
    ), f"Failure reason not found for request ticket {dead_letter_ticket_name}"

    clean_up_ticket(f"{DEAD_LETTER_QUEUE_FOLDER}/{dead_letter_ticket_name}")

    LOGGER.info("Tests passed for querying tickets on the dead letter queue.")


def main():
    job_requester_object = JobRequester()
    request_ticket_prefix = f"testing-0_{REQUEST_TICKET_TIME}"
    # create identifier for the request ticket
    request_identifier = Message(
        SQS_RETURN_QUEUE_URL, BUCKET_NAME, f"{request_ticket_prefix}.json", TEST_ECR_URI, REQUEST_TICKET_TIME
    )
    test_query_and_cancel_queuing_tickets(job_requester_object, f"{request_ticket_prefix}.json", request_identifier)

    # naming convention of in-progress pool tickets: {request ticket name}#{num of instances}-{status}.json
    in_progress_ticket_name = f"{request_ticket_prefix}#1-running.json"
    test_query_in_progress_tickets(job_requester_object, in_progress_ticket_name, request_identifier)

    # naming convention of in-progress pool tickets: {request ticket name}-{failure reason}.json
    dead_letter_ticket_name = f"{request_ticket_prefix}-timeout.json"
    test_query_dead_letter_tickets(job_requester_object, dead_letter_ticket_name, request_identifier)

    LOGGER.info("Tests passed.")


if __name__ == "__main__":
    main()
