import boto3
import constants
import random

class Metrics(object):
    def __init__(self,context="DEV",region="us-west-2", namespace="dlc-metrics-sk"):
        self.client = boto3.Session(region_name=region).client('cloudwatch')
        self.context = context
        self.namespace=namespace
    
    def push(self, name, unit, value, info):

        dimensions = [{'Name': 'BuildContext',
                'Value': self.context }]
    
        for k in info:
            dimensions.append({'Name': k, 'Value': info[k]})
    
        try:
            response = self.client.put_metric_data(
                MetricData = [
                    {
                        'MetricName': name,
                        'Dimensions': dimensions,
                        'Unit': unit,
                        'Value': value
                    },
                ],
                Namespace = self.namespace
                )
        except Exception as e:
            raise Exception(str(e))

        return response
    
    def push_image_metrics(self, image):
        info = {"framework": image.framework,
            "version": image.version,
            "device_type": image.device_type,
            "python_version": image.python_version,
            "image_type": image.image_type
            }
        if image.build_status == constants.NOT_BUILT:
            return None
        build_time = (image.summary["endtime"] - image.summary["starttime"]).seconds
        build_status = image.build_status

        self.push("build_time", "Seconds", build_time, info)
        self.push("build_status", "None", build_status, info)

        if image.build_status == constants.FAIL:
            image_size = image.summary["image_size"]
            self.push("image_size", "Bytes", image_size, info)
