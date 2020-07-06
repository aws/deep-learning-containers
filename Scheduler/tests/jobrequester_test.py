import concurrent.futures
import json

import boto3
from job_requester.requester import JobRequester


def testrunner():
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
    response_list = []
    input_list = []

    # creating unique image names and build_context strings
    for i in range(threads):
        input_list.append((f"image{str(i)}", f"build_context{str(i)}"))

    # sending requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(request_object.send_request, x, y) for (x, y) in input_list]

    print("Created tickets......")
    for future in futures:
        res = future.result()
        print(res)
        response_list.append(res)
    print("\n")

    # sending messages to SQS, imitating the response log from Job Executor
    sqs_url = request_object.sqs_queue
    sqs_client = boto3.client("sqs")
    prefix = sqs_url.split("/")[-1]
    for m in range(threads):
        content = {}
        content["TICKET_NAME"] = f"{prefix}-{str(m)}.json"
        content["TEXT"] = str(m)
        sqs_client.send_message(QueueUrl=sqs_url, MessageBody=json.dumps(content))

    # receiving logs
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        logs = [executor.submit(request_object.receive_logs, identifier) for identifier in response_list]

    print("\nPolling for messages.....")
    for log in logs:
        print(log.result())

    print("\nEntire Log:")
    print(request_object.logs)

    # clean up test artifacts
    S3 = boto3.client("s3")
    ticket_names = [item.ticket_name for item in response_list]
    for name in ticket_names:
        S3.delete_object(Bucket=request_object.s3_ticket_bucket, Key=name)
    request_object.clean_up()

    print("\nClean up done.")


if __name__ == "__main__":
    testrunner()
