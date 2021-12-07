import io

import grpc
import gzip
import numpy as np
import tensorflow as tf

from google.protobuf.json_format import MessageToJson
from PIL import Image
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

prediction_services = {}
compression_algo = gzip


def handler(data, context):
    f = data.read()
    f = io.BytesIO(f)
    image = Image.open(f).convert('RGB')
    batch_size = 1
    image = np.asarray(image.resize((224, 224)))
    image = np.concatenate([image[np.newaxis, :, :]] * batch_size)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = context.model_name
    request.model_spec.signature_name = 'serving_default'
    request.inputs['images'].CopyFrom(
        tf.compat.v1.make_tensor_proto(image, shape=image.shape, dtype=tf.float32))

    # Call Predict gRPC service
    result = get_prediction_service(context).Predict(request, 60.0)
    print("Returning the response for grpc port: {}".format(context.grpc_port))

    # Return response
    json_obj = MessageToJson(result)
    return json_obj, "application/json"


def get_prediction_service(context):
    # global prediction_service
    if context.grpc_port not in prediction_services:
        channel = grpc.insecure_channel("localhost:{}".format(context.grpc_port))
        prediction_services[context.grpc_port] = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return prediction_services[context.grpc_port]
