import json


class Message:
    def __init__(self, sqs_queue_url, ticket_bucket, ticket_name, image, request_time):
        self.sqs_url = sqs_queue_url
        self.image = image
        self.request_time = request_time
        self.ticket_bucket = ticket_bucket
        self.ticket_name = ticket_name

        data_set = {"SQS_URL": sqs_queue_url, "S3_bucket": ticket_bucket, "S3_ticket_name": ticket_name}
        self.data = json.dumps(data_set)

    def __str__(self):
        return self.data
