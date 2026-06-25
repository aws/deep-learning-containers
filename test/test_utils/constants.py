DEFAULT_REGION = "us-west-2"
EC2_INSTANCE_ROLE_NAME = "ec2TestInstanceRole"
SAGEMAKER_ROLE = "SageMakerRole"
# https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html
# Tests should pick the AMI matching their container's CUDA major version.
# AL2 AMI: NVIDIA driver 550 — for CUDA 12.x containers.
INFERENCE_AMI_VERSION_CU12 = "al2-ami-sagemaker-inference-gpu-3-1"
# AL2023 AMI: NVIDIA driver 580 — for CUDA 13.x containers.
INFERENCE_AMI_VERSION_CU13 = "al2023-ami-sagemaker-inference-gpu-4-1"
# Default for tests
INFERENCE_AMI_VERSION = INFERENCE_AMI_VERSION_CU13
