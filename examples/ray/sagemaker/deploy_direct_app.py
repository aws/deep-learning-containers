from sagemaker.model import Model
from sagemaker.predictor import Predictor

model = Model(
    image_uri="{{ images.latest_ray_sagemaker_gpu }}",
    role="arn:aws:iam::<ACCOUNT>:role/SageMakerExecutionRole",
    model_data="s3://<BUCKET>/models/mnist/model.tar.gz",
    predictor_cls=Predictor,
    env={"SM_RAYSERVE_APP": "deployment:app", "RAYSERVE_NUM_GPUS": "1"},
)

predictor = model.deploy(
    instance_type="ml.g5.xlarge",
    initial_instance_count=1,
    endpoint_name="ray-serve-mnist",
    inference_ami_version="al2-ami-sagemaker-inference-gpu-3-1",
    wait=True,
)
