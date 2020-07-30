import concurrent.futures
import logging
import os
import sys


import boto3

import log_return

from job_requester import JobRequester


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

TEST_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.2.0-gpu-py37-cu101-ubuntu18.04"
SAMPLE_XML_MESSAGE = "<note><to>Sample</to><from>XML</from><heading>Report</heading><body>Hello World!</body></note>"
SAMPLE_CB_ARN = "arn:aws:codebuild:us-west-2:754106851545:build/DLCTestJobExecutor:894c9690-f6dc-4a15-b4b8-b9f2ddc51ea9"


def test_requester():
    """
    Tests the send_request and receive_logs functions of the Job Requester package.
    How tests are executed:
    - create one Job Requester object, and multiple threads. Perform send_request with the Job Requester object in
      each of these threads.
    - send messages to the SQS queue that the Job Requester object created, to imitate the response logs received back
      from the Job Executor.
    - In each of the threads, perform receive_logs to receive the log correspond to the send_request earlier.
    """
    threads = 10
    request_object = JobRequester()
    identifiers_list = []
    input_list = []

    # creating unique image names and build_context strings
    for _ in range(threads):
        input_list.append((TEST_IMAGE, "PR", 3))

    # sending requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(request_object.send_request, x, y, z) for (x, y, z) in input_list]

    print("Created tickets......")
    for future in futures:
        res = future.result()
        print(res)
        identifiers_list.append(res)
    print("\n")

    # create sample xml report files
    image_tag = TEST_IMAGE.split(":")[-1]
    report_path = os.path.join(os.getcwd(), f"{image_tag}.xml")
    with open(report_path, "w") as report:
        report.write(SAMPLE_XML_MESSAGE)

    os.environ["CODEBUILD_BUILD_ARN"] = SAMPLE_CB_ARN
    for identifier in identifiers_list:
        os.environ["TICKET_KEY"] = f"folder/{identifier.ticket_name}"
        log_return.update_pool("completed", identifier.instance_type, 3, identifier.job_type, report_path)

    # receiving logs
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        logs = [executor.submit(request_object.receive_logs, identifier) for identifier in identifiers_list]

    LOGGER.info("Receiving logs...")
    for log in logs:
        assert "XML_REPORT" in log.result(), f"XML Report not found as part of the returned log message."

    # clean up test artifacts
    S3 = boto3.client("s3")
    ticket_names = [item.ticket_name for item in identifiers_list]
    for name in ticket_names:
        S3.delete_object(Bucket=request_object.s3_ticket_bucket, Key=name)

    LOGGER.info("Tests passed.")


if __name__ == "__main__":
    test_requester()
