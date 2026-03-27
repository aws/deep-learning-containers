import json

from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

predictor = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/ray:serve-ml-sagemaker-cuda",
    role="arn:aws:iam::<ACCOUNT>:role/SageMakerExecutionRole",
    model_data="s3://<BUCKET>/models/nlp-sentiment/model.tar.gz",
    predictor_cls=Predictor,
).deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
    endpoint_name="ray-serve-nlp",
    serializer=JSONSerializer(),
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    wait=True,
)

response = predictor.predict({"text": "I love this so much, best purchase ever!"})
result = json.loads(response)  # predictor.predict() returns raw bytes
# {"predictions": [{"label": "POSITIVE", "score": 0.9991}]}
