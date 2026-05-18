from sagemaker.model import Model
from sagemaker.predictor import Predictor

# RAYSERVE_NUM_GPUS is a user-defined convention used by this example's
# deployment.py to set ray_actor_options.num_gpus on the Serve deployment.
# The DLC image does not read it; it's only meaningful if your deployment.py
# reads it via os.getenv("RAYSERVE_NUM_GPUS").
model = Model(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/ray:serve-ml-sagemaker-cuda",
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
