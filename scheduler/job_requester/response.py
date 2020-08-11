import json


class Message:
    def __init__(self, ticket_bucket, ticket_name, image, instance_type, job_type, request_time):
        self.image = image
        self.request_time = request_time
        self.ticket_bucket = ticket_bucket
        self.ticket_name = ticket_name
        self.instance_type = instance_type
        self.job_type = job_type

        data_set = {
            "instance_type": instance_type,
            "job_type": job_type,
            "S3_bucket": ticket_bucket,
            "S3_ticket_name": ticket_name,
        }
        self.data = json.dumps(data_set)

    def __str__(self):
        return self.data
