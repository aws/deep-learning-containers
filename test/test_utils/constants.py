DEFAULT_REGION = "us-west-2"
SAGEMAKER_ROLE = "SageMakerRole"
# https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html
INFERENCE_AMI_VERSION = "al2-ami-sagemaker-inference-gpu-3-1"

FRAMEWORK_MODULE_MAP = {
    "pytorch": "torch",
    "tensorflow": "tensorflow",
    "vllm": "vllm",
    "sglang": "sglang",
    "base": None,
}
