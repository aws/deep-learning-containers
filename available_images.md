# Framework Support Policy

The framework support policy is live on the [DLC](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/support-policy.html) dev guides.

Anaconda shifted to a commercial licensing model for certain users. Actively maintained DLCs have been migrated to the publicly available open-source version of Conda (conda-forge) from the Anaconda channel.

> Warning: If you are actively using Anaconda to install and manage your packages and their dependencies in a DLC that is no longer actively maintained, you are responsible for complying with the governing license from the Anaconda Repository, if you determine that the terms apply to you. Alternatively, you can migrate to one of the currently-supported DLCs listed in the Supported Frameworks table or you can install packages using conda-forge as a source.

# Available Deep Learning Containers Images

The following table lists the Docker image URLs that will be used by Amazon ECS in task definitions. Replace the `<repository-name>` and `<image-tag>` values based on your desired container.

Once you've selected your desired Deep Learning Containers image, continue with one of the following tutorials:

-   To run training and inference on Deep Learning Containers for Amazon EC2 using PyTorch and TensorFlow, see [Amazon EC2 Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2.html)

-   To run training and inference on Deep Learning Containers for Amazon ECS using PyTorch and TensorFlow, see [Amazon ECS tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs.html)

-   Deep Learning Containers for Amazon EKS offer CPU, GPU, and distributed GPU-based training, as well as CPU and GPU-based inference. To run training and inference on Deep Learning Containers for Amazon EKS using PyTorch, and TensorFlow, see [Amazon EKS Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-eks.html)

-   For information on security in Deep Learning Containers, see [Security in AWS Deep Learning Containers](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/security.html)

-   For a list of the latest Deep Learning Containers release notes, see [Release Notes for Deep Learning Containers](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-release-notes.html)


Deep Learning Containers Docker Images are available in the following regions:

| Region 					               | Code 				        |General Container	|Neuron Container	| Example URL																				                                                           |
|----------------------------|------------------|-------------------|-------------------|-------------------------------------------------------------------------------------------|
| US East (Ohio)				         | us-east-2			     |Available 			|Available			| 763104351884.dkr.ecr.us-east-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| US East (N. Virginia)		    | us-east-1			     |Available 			|Available			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| US West (N. California)	   | us-west-1			     |Available 			|None				| 763104351884.dkr.ecr.us-west-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| US West (Oregon)			        | us-west-2			     |Available 			|Available			| 763104351884.dkr.ecr.us-west-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| Africa (Cape Town)			      | af-south-1			    |Available			|None				| 626614931356.dkr.ecr.af-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| Asia Pacific (Hong Kong)	  | ap-east-1			     |Available 			|None				| 871362719292.dkr.ecr.ap-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| Asia Pacific (Hyderabad)		 | ap-south-2			    |Available 			|None			| 772153158452.dkr.ecr.ap-south-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| Asia Pacific (Jakarta)		   | ap-southeast-3 	 |Available 			|None			| 907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Malaysia)		 | ap-southeast-5 	 |Available 			|None			| 550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Melbourne)		 | ap-southeast-4 	 |Available 			|None			| 457447274322.dkr.ecr.ap-southeast-4.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Mumbai)		    | ap-south-1			    |Available 			|Available			| 763104351884.dkr.ecr.ap-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| Asia Pacific (Osaka)		     | ap-northeast-3		 |Available 			|None				| 364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Seoul)		     | ap-northeast-2		 |Available 			|None				| 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Singapore)	  | ap-southeast-1		 |Available 			|Available			| 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Sydney)		    | ap-southeast-2 	 |Available 			|Available			| 763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Taipei)		    | ap-east-2 	  |Available 			|Available			| 763104351884.dkr.ecr.ap-east-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Thailand)        | ap-southeast-7    |Available             |None           | 590183813437.dkr.ecr.ap-southeast-7.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Tokyo)		     | ap-northeast-1		 |Available 			|Available			| 763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Canada (Central)			        | ca-central-1		   |Available 			|None				| 763104351884.dkr.ecr.ca-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		     |
| Canada (Calgary)			        | ca-west-1		   |Available 			|None				| 204538143572.dkr.ecr.ca-west-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		     |
| EU (Frankfurt) 			         | eu-central-1		   |Available 			|Available			| 763104351884.dkr.ecr.eu-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		     |
| EU (Ireland) 				          | eu-west-1			     |Available 			|Available			| 763104351884.dkr.ecr.eu-west-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| EU (London) 				           | eu-west-2			     |Available 			|None				| 763104351884.dkr.ecr.eu-west-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| EU (Milan)					            | eu-south-1			    |Available			|None				| 692866216735.dkr.ecr.eu-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| EU (Paris) 				            | eu-west-3			     |Available 			|Available			| 763104351884.dkr.ecr.eu-west-3.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| EU (Spain)					            | eu-south-2			    |Available			|None				| 503227376785.dkr.ecr.eu-south-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| EU (Stockholm) 			         | eu-north-1			    |Available 			|None				| 763104351884.dkr.ecr.eu-north-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| EU (Zurich) 			            | eu-central-2		   |Available 			|None			| 380420809688.dkr.ecr.eu-central-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>		     |
| Israel (Tel Aviv) 			      | il-central-1			  |Available 			|None				| 780543022126.dkr.ecr.il-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			    |
| Mexico (Central)                    | mx-central-1            |Available          |None               | 637423239942.dkr.ecr.mx-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| Middle East (Bahrain) 		   | me-south-1			    |Available 			|None				| 217643126080.dkr.ecr.me-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| Middle East (UAE)		        | me-central-1 	   |Available 			|None			| 914824155844.dkr.ecr.me-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		     |
| South America (Sao Paulo)	 | sa-east-1			     |Available 			|Available			| 763104351884.dkr.ecr.sa-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| China (Beijing)			         | cn-north-1			    |Available 			|None				| 727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/&lt;repository-name>:&lt;image-tag>		    |
| China (Ningxia)			         | cn-northwest-1		 |Available 			|None				| 727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/&lt;repository-name>:&lt;image-tag>	 |

ECR is a regional service and the Image table contains the URLs for
**us-east-1** images. To pull from one of the regions mentioned
previously, insert the region in the repository URL following this
example:


    763104351884.dkr.ecr.<region>.amazonaws.com/tensorflow-training:2.9.1-gpu-py39-cu112-ubuntu20.04-ec2

**Important**

You must login to access the DLC image repository before pulling
the image. Ensure your CLI is up to date using the steps in [Installing the current AWS CLI Version](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv1.html#install-tool-bundled)
    Then, specify your region and its corresponding ECR Registry from
    the previous table in the following command:



    aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

You can then pull these Docker images from ECR by running:



    docker pull <name of container image>

DLC Available Image User Guide
============================

To use the following tables, select your desired framework, the kind of job you're starting, and your desired Python version. Your
job type is either ``training``, ``inference`` or ``general``. Your Python version is
either ``py37``, ``py38``, ``py39``, ``py310``, ``py311`` or ``py312`` depending on availability. Plug this information into the replaceable portions of the URL as shown in the example URL.

You can pin your version by adding the version tag to your URL as follows:

    763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.4.1-cpu-py37-ubuntu18.04-v1.0


Base Containers 
============================

| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	| Example URL																						                                                                         |
|-------------------|-----------|---------------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------------|
| CUDA 12.8 + EFA   |General	|No			|GPU 		| 3.12 (py312)			| 763104351884.dkr.ecr.us-west-2.amazonaws.com/base:12.8.1-gpu-py312-cu128-ubuntu24.04-ec2        |


EC2 vLLM Containers (EC2, ECS, and EKS support only)
============================

| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	| Example URL																						                                                                         |
|-------------------|-----------|---------------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------------|
| vLLM 0.9 + EFA   |General	|No			|GPU 		| 3.12 (py312)			| 763104351884.dkr.ecr.us-west-2.amazonaws.com/vllm:0.9-gpu-py312-ec2        |


EC2 Framework Containers (EC2, ECS, and EKS support only)
============================

| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	| Example URL																						                                                                         |
|-------------------|-----------|---------------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------------|
| PyTorch 2.7.1     |training	|No			|CPU 		| 3.12 (py312)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-cpu-py312-ubuntu22.04-ec2             |
| PyTorch 2.7.1     |training	|No			|GPU 		| 3.12 (py312)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-ec2       |
| PyTorch 2.6.0     |inference	|No			|CPU 		| 3.12 (py312)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-ec2            |
| PyTorch 2.6.0     |inference	|No			|GPU 		| 3.12 (py312)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-ec2      |
| TensorFlow 2.18.0 |training	|No			|CPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.18.0-cpu-py310-ubuntu22.04-ec2		      |
| TensorFlow 2.18.0 |training	|No			|GPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.18.0-gpu-py310-cu125-ubuntu22.04-ec2	 |
| TensorFlow 2.18.0 |inference	|No			|CPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.18.0-cpu-py310-ubuntu20.04-ec2		      |
| TensorFlow 2.18.0 |inference	|No			|GPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.18.0-gpu-py310-cu122-ubuntu20.04-ec2	 |


SageMaker Framework Containers (SM support only)
============================

| Framework         | Job Type  | Horovod Options | CPU/GPU   | Python Version Options | Example URL																						                                                                              |
|-------------------|-----------|-----------------|-----------|------------------------|----------------------------------------------------------------------------------------------------------------|
| PyTorch 2.7.1     | training	| No			  | CPU 	  | 3.12 (py312)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-cpu-py312-ubuntu22.04-sagemaker           |
| PyTorch 2.7.1     | training	| No			  | GPU 	  | 3.12 (py312)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.7.1-gpu-py312-cu128-ubuntu22.04-sagemaker     |
| PyTorch 2.6.0     | inference	| No			  | CPU 	  | 3.12 (py312)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-cpu-py312-ubuntu22.04-sagemaker           |
| PyTorch 2.6.0     | inference	| No			  | GPU 	  | 3.12 (py312)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker     |
| TensorFlow 2.19.0 | training  | No			  | CPU 	  | 3.12 (py312)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.19.0-cpu-py312-ubuntu22.04-sagemaker		|
| TensorFlow 2.19.0 | training  | No			  | GPU 	  | 3.12 (py312)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.19.0-gpu-py312-cu125-ubuntu22.04-sagemaker  |
| TensorFlow 2.19.0 | inference | No			  | CPU 	  | 3.12 (py312)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.19.0-cpu-py312-ubuntu22.04-sagemaker	      |
| TensorFlow 2.19.0 | inference | No			  | GPU 	  | 3.12 (py312)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.19.0-gpu-py312-cu122-ubuntu22.04-sagemaker |


EC2 Framework ARM64/Graviton Containers (EC2, ECS, and EKS support only)
============================
Important note: Starting with PyTorch 2.5, we are changing the name of Graviton DLCs to ARM64 DLCs in order to generalize the usage. For example, the ECR repository name is now "pytorch-inference-arm64"
instead of "pytorch-inference-graviton". Graviton DLCs and ARM64 DLCs are functionally equivalent.

| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	| Example URL																						                                                                             |
|-------------------|-----------|---------------|-----------|-----------------------|---------------------------------------------------------------------------------------------------------------|
| PyTorch 2.7.0     |training	|No			|GPU 		| 3.12 (py312)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training-arm64:2.7.0-gpu-py312-cu128-ubuntu22.04-ec2		     |
| PyTorch 2.6.0     |inference	|No			|CPU 		| 3.12 (py312)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-arm64:2.6.0-cpu-py312-ubuntu22.04-ec2		     |
| PyTorch 2.6.0     |inference	|No			|GPU 		| 3.12 (py312)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-arm64:2.6.0-gpu-py312-cu124-ubuntu22.04-ec2		     |
| TensorFlow 2.18.0 |inference	|No			|CPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-arm64:2.18.0-cpu-py310-ubuntu20.04-ec2		 |


SageMaker Framework ARM64/Graviton Containers (SM support only)
============================
Important note: Starting with PyTorch 2.5, we are changing the name of Graviton DLCs to ARM64 DLCs in order to generalize the usage. For example, the ECR repository name is now "pytorch-inference-arm64"
instead of "pytorch-inference-graviton". Graviton DLCs and ARM64 DLCs are functionally equivalent.

| Framework         | Job Type	| Horovod Options| CPU/GPU | Python Version Options	| Example URL																						                                                                                 |
|-------------------|-----------|----------------|---------|------------------------|-------------------------------------------------------------------------------------------------------------------|
| PyTorch 2.6.0     | inference | No			 | CPU 	   | 3.12 (py312)			    | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-arm64:2.6.0-cpu-py312-ubuntu22.04-sagemaker     |
| TensorFlow 2.18.0 | inference | No			 | CPU 	   | 3.10 (py310)			    | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-arm64:2.18.0-cpu-py310-ubuntu20.04-sagemaker |


NVIDIA Triton Inference Containers (SM support only)
============================
**Versions 23.12 and onwards**:
1. Starting version 23.12, onwards, the sagemaker tritonserver is available in a different set of accounts than the previous ones. The new accounts are now the same as other DLCs listed on this page at the top, making it easier to switch containers going forward. These accounts are listed in section '# Available Deep Learning Containers Images' above.
2. They can now be obtained programmatically from the sagemaker python sdk as:
```
from sagemaker import image_uris

triton_framework = "sagemaker-tritonserver"
region="us-west-2"
instance_type="ml.g5.12xlarge"

available_versions = list(image_uris.config_for_framework(triton_framework)['versions'].keys())
image_uri = image_uris.retrieve(framework=triton_framework, region=region, instance_type=instance_type, version=available_versions[0])
```
3. Available versions: `25.04`, `24.09`, `24.05`, `24.03`, `24.01`, `23.12`

The Sagemaker Triton inference containers are built on top of the NGC containers with SageMaker support. To identify the python version and versions for other packages please refer to the corresponding official release notes for the specific version here: https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html

**Versions prior to 23.12**:

1. For versions prior to 23.12, the following 23.`<XY>` versions are available: `23.01, 23.02, 23.03, 23.05, 23.06, 23.07, 23.08, 23.09, 23.10`.
2. For versions of the 22.`<XY>` series, the following are available: `22.05, 22.07, 22.08, 22.09, 22.10, 22.12`
3. For versions of the 21.`<XY>` series, the following are available: `21.08`
4. The following example notebook demonstrates the account_id_map to obtain the account for versions prior to r23.12:
https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-triton/resnet50/triton_resnet50.ipynb

| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	|Example URL																						|
|-------------------|-----------|---------------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|NVIDIA Triton Inference Server 23.`<XY>`    |inference	|No			|GPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:23.`<XY>`-py3		|
|NVIDIA Triton Inference Server 23.`<XY>`    |inference	|No			|CPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:23.`<XY>`-py3-cpu		|

**Note**:
1. SageMaker Triton Inference Container does not support Tensorflow1 as of version 23.05 onwards, as upstream Triton container does not support Tensorflow(v1) native backend from version 23.04 onwards.
2. SageMaker Triton Inference Container does not ship with the FasterTransformer(FT) backend from version 23.06 onwards since the upstream FT library is undergoing re-structuring. It was previously available from versions v22.12 - v23.05, experimentally.


Large Model Inference Containers
===============================
Starting LMI V10 (0.28.0), we are changing the name from LMI DeepSpeed DLC to LMI (LargeModelInference). As part of this change, we have decided to discontinue integration with DeepSpeed library into the container. You can continue to use vLLM or LMI-dist Library with the LMI container. If you plan to use DeepSpeed Library, please follow the steps [here](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/lmi/announcements/deepspeed-deprecation.md) or use LMI V9 (0.27.0).

| Framework                                                                                                                   | Job Type  | Accelerator | Python Version Options | Example URL                                                                               |
|-----------------------------------------------------------------------------------------------------------------------------|-----------|-------------|------------------------|-------------------------------------------------------------------------------------------|
| DJLServing 0.33.0 with LMI Dist 15.0.0, vLLM 0.8.4, HuggingFace Transformers 4.51.3, and HuggingFace Accelerate 1.0.1 | inference | GPU | 3.12 (py312) | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128 |
| DJLServing 0.33.0 with TensorRT-LLM 0.21.0rc1, HuggingFace Transformers 4.51.3, and HuggingFace Accelerate 1.0.1 | inference | GPU | 3.12 (py312)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-tensorrtllm0.21.0-cu128 |
| DJLServing 0.32.0 with LMI Dist 13.0.0, vLLM 0.7.1, HuggingFace Transformers 4.45.2, and HuggingFace Accelerate 1.0.1 | inference | GPU         | 3.11 (py311)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.32.0-lmi14.0.0-cu126         |
| DJLServing 0.32.0 with TensorRT-LLM 0.12.0, HuggingFace Transformers 4.44.2, and HuggingFace Accelerate 0.32.1              | inference | GPU         | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.32.0-tensorrtllm0.12.0-cu125 |
| DJLServing 0.31.0 with LMI Dist 13.0.0, vLLM 0.6.3.post1, HuggingFace Transformers 4.45.2, and HuggingFace Accelerate 1.0.1 | inference | GPU         | 3.11 (py311)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.31.0-lmi13.0.0-cu124         |
| DJLServing 0.30.0 with LMI Dist 12.0.0, vLLM 0.6.2, HuggingFace Transformers 4.45.2, and HuggingFace Accelerate 1.0.1       | inference | GPU         | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.30.0-lmi12.0.0-cu124         |
| DJLServing 0.30.0 with TensorRT-LLM 0.12.0, HuggingFace Transformers 4.44.2, and HuggingFace Accelerate 0.33.0              | inference | GPU         | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.30.0-tensorrtllm0.12.0-cu125 |
| DJLServing 0.30.0 with Neuron SDK 2.20.1, TransformersNeuronX 0.12.313, and HuggingFace Transformers 4.45.2                 | inference | Neuron      | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.30.0-neuronx-sdk2.20.1       |
| DJLServing 0.29.0 with TensorRT-LLM 0.11.0, HuggingFace Transformers 4.42.4, and HuggingFace Accelerate 0.32.1              | inference | GPU         | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.29.0-tensorrtllm0.11.0-cu124 |
| DJLServing 0.29.0 with LMI Dist 11.0.0, HuggingFace Transformers 4.43.2, HuggingFace Accelerate 0.32.1                      | inference | GPU         | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.29.0-lmi11.0.0-cu124         |
| DJLServing 0.29.0 with Neuron SDK 2.19.1, TransformersNeuronX 0.11.351 and HuggingFace Transformers 4.43.1                  | inference | Neuron      | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.29.0-neuronx-sdk2.19.1       |
| DJLServing 0.28.0 with TensorRT-LLM 0.9.0, HuggingFace Transformers 4.40.0, and HuggingFace Accelerate 0.29.3               | inference | GPU         | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-tensorrtllm0.9.0-cu122  |
| DJLServing 0.28.0 with LMI Dist 0.10.0, HuggingFace Transformers 4.41.1, HuggingFace Accelerate 0.30.1                      | inference | GPU         | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-lmi10.0.0-cu124         |
| DJLServing 0.28.0 with Neuron SDK 2.18.2, TransformersNeuronX 0.10.0.360 and HuggingFace Transformers 4.36.2                | inference | Neuron      | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-neuronx-sdk2.18.2       |

DJL CPU Full Inference Containers
===============================

| Framework         | Job Type  | CPU/GPU | Python Version Options | Example URL                                                                |
|-------------------|-----------|---------|------------------------|----------------------------------------------------------------------------|
| DJLServing 0.29.0 | inference | CPU     | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.29.0-cpu-full |
| DJLServing 0.28.0 | inference | CPU     | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.28.0-cpu-full |
| DJLServing 0.27.0 | inference | CPU     | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.27.0-cpu-full |

AutoGluon Training Containers
===============================

| Framework       | AutoGluon Version  | Job Type | CPU/GPU | Python Version Options | Example URL                                                                                      |
|-----------------|--------------------|----------|---------|------------------------|--------------------------------------------------------------------------------------------------|
| AutoGluon 1.3.1 | 1.3.1              | training | GPU     | 3.11 (py311)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:1.3.1-gpu-py311-cu124-ubuntu22.04 |
| AutoGluon 1.3.1 | 1.3.1              | training | CPU     | 3.11 (py311)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:1.3.1-cpu-py311-ubuntu22.04 |
AutoGluon Inference Containers
===============================

| Framework       | AutoGluon Version  | Job Type  | CPU/GPU | Python Version Options | Example URL                                                                                       |
|-----------------|--------------------|-----------|---------|------------------------|---------------------------------------------------------------------------------------------------|
| AutoGluon 1.3.1 | 1.3.1              | inference | GPU     | 3.11 (py311)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-inference:1.3.1-gpu-py311-cu124-ubuntu22.04 |
| AutoGluon 1.3.1 | 1.3.1              | inference | CPU     | 3.11 (py311)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-inference:1.3.1-cpu-py311-ubuntu22.04 |
HuggingFace Training Containers
===============================

| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL                                                                                                                        |
|-----------------------------------------------|-----------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------|
|PyTorch 2.1.0 with HuggingFace transformers    |training	|GPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04     |
|PyTorch 2.0.0 with HuggingFace transformers    |training	|GPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04     |
|PyTorch 1.13.1 with HuggingFace transformers   |training	|GPU 		| 3.9 (py39)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04     |
|TensorFlow 2.6.3 with HuggingFace transformers |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-training:2.6.3-transformers4.17.0-gpu-py38-cu112-ubuntu20.04 	|


HuggingFace Inference Containers
===============================

| Framework                                        |Job Type	|CPU/GPU 	|Python Version Options	|Example URL                                                                                                                        |
|--------------------------------------------------|------------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------|
|PyTorch 2.1.0 with HuggingFace transformers       |inference	|CPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-cpu-py310-ubuntu22.04		    |
|PyTorch 2.1.0 with HuggingFace transformers       |inference	|GPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.1.0-transformers4.37.0-gpu-py310-cu118-ubuntu20.04    |
|PyTorch 2.0.0 with HuggingFace transformers       |inference	|CPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04		    |
|PyTorch 2.0.0 with HuggingFace transformers       |inference	|GPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04    |
|PyTorch 1.13.1 with HuggingFace transformers      |inference	|CPU 		| 3.9 (py39)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04		    |
|PyTorch 1.13.1 with HuggingFace transformers      |inference	|GPU 		| 3.9 (py39)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04    |
|TensorFlow 2.11.1 with HuggingFace transformers   |inference	|CPU 		| 3.9 (py39)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.11.1-transformers4.26.0-cpu-py39-ubuntu20.04		|
|TensorFlow 2.11.1 with HuggingFace transformers   |inference	|GPU 		| 3.9 (py39)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.11.1-transformers4.26.0-gpu-py39-cu112-ubuntu20.04	|

HuggingFace Text Generation Inference (TGI) Containers
===============================

Please refer to the following pages to view all available versions and tags for GPU and NeuronX containers:
* [GPU Release Page](https://github.com/aws/deep-learning-containers/releases?q=tgi+AND+gpu&expanded=true)
* [NeuronX Release Page](https://github.com/aws/deep-learning-containers/releases?q=tgi+AND+neuronx&expanded=true)

HuggingFace Neuron Inference Containers
===============================

|Framework                                                         |Neuron SDK Version |Job Type   |Supported EC2 Instance Type |Python Version Options |Example URL                                                                                                                                   |
|------------------------------------------------------------------|-------------------|-----------|----------------------------|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
|PyTorch 1.10.2 with Neuron Inference and HuggingFace transformers |Neuron 1.19.1      |inference  |inf1                        |3.7 (py37)             |763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuron:1.10.2-transformers4.20.1-neuron-py37-sdk1.19.1-ubuntu18.04 |
|PyTorch 1.13.1 with NeuronX Inference and HuggingFace transformers |Neuron 2.15.0      |inference  |inf2/trn1                        |3.10 (py310)             |763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuronx:1.13.1-transformers4.34.1-neuronx-py310-sdk2.15.0-ubuntu20.04 |
|PyTorch 2.1.2 with NeuronX Inference and HuggingFace transformers |Neuron 2.18.0      |inference  |inf2/trn1                        |3.10 (py310)             |763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuronx:2.1.2-transformers4.36.2-neuronx-py310-sdk2.18.0-ubuntu20.04 |

HuggingFace Neuron Training Containers
===============================

|Framework                                                         |Neuron SDK Version |Job Type   |Supported EC2 Instance Type |Python Version Options |Example URL                                                                                                                                   |
|------------------------------------------------------------------|-------------------|-----------|----------------------------|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
|PyTorch 1.13.1 with NeuronX Training and HuggingFace transformers |Neuron 2.18.0      |training  |trn1                        |3.10 (py310)             |763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-neuronx:1.13.1-transformers4.36.2-neuronx-py310-sdk2.18.0-ubuntu20.04 |

StabilityAI Inference Containers
===============================

| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL                                                                                                                        |
|-----------------------------------------------|-----------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------|
|PyTorch 2.0.1 with StabilityAI SGM    |inference	|GPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/stabilityai-pytorch-inference:2.0.1-sgm0.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker     |

SageMaker Training Compiler Containers
===============================

| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL																						|
|-----------------------------------------------|-----------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
| TensorFlow 2.10.0                              |training   |GPU        | 3.9 (py39)            | 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.10.0-gpu-py39-cu112-ubuntu20.04-sagemaker     |
|PyTorch 1.13.1 with SageMaker Training Compiler    |training	|GPU 		| 3.9 (py39 )			|763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-trcomp-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker     |
|PyTorch 1.11.0 with HuggingFace transformers 4.21.1 and SageMaker Training Compiler    |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-trcomp-training:1.11.0-transformers4.21.1-gpu-py38-cu113-ubuntu20.04      |


Neuron Containers
=================

Note: Starting from Neuron SDK 2.17.0, Dockerfiles for PyTorch Neuron Containers can be accessed at https://github.com/aws-neuron/deep-learning-containers.

| Framework                                                                                                                                          | Neuron Package                                           | Neuron SDK Version | Job Type  | Supported EC2 Instance Types | Python Version Options | Example URL                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|--------------------|-----------|------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------|
| [PyTorch 2.6.0](https://github.com/aws-neuron/deep-learning-containers/blob/2.23.0/docker/pytorch/inference/2.6.0/Dockerfile.neuronx)              | torch-neuronx, transformers-neuronx, neuronx_distributed, neuronx_distributed_inference | Neuron 2.23.0      | inference | trn1,trn2,inf2                    | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.6.0-neuronx-py310-sdk2.23.0-ubuntu22.04     |
| [PyTorch 2.6.0](https://github.com/aws-neuron/deep-learning-containers/blob/2.23.0/docker/pytorch/training/2.6.0/Dockerfile.neuronx)              | torch-neuronx, transformers-neuronx, neuronx_distributed, neuronx_distributed_training | Neuron 2.23.0      | training | trn1,trn2,inf2                    | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.6.0-neuronx-py310-sdk2.23.0-ubuntu22.04     |
| [PyTorch 2.5.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.22.0/docker/pytorch/inference/2.5.1/Dockerfile.neuronx)              | torch-neuronx, transformers-neuronx, neuronx_distributed, neuronx_distributed_inference | Neuron 2.22.0      | inference | trn1,trn2,inf2                    | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.5.1-neuronx-py310-sdk2.22.0-ubuntu22.04     |
| [PyTorch 2.5.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.22.0/docker/pytorch/training/2.5.1/Dockerfile.neuronx)              | torch-neuronx, transformers-neuronx, neuronx_distributed, neuronx_distributed_training | Neuron 2.22.0      | training | trn1,trn2,inf2                    | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.5.1-neuronx-py310-sdk2.22.0-ubuntu22.04     |
| [PyTorch 2.1.2](https://github.com/aws-neuron/deep-learning-containers/blob/2.20.2/docker/pytorch/inference/2.1.2/Dockerfile.neuronx)              | torch-neuronx, transformers-neuronx, neuronx_distributed | Neuron 2.20.2      | inference | trn1,inf2                    | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.20.2-ubuntu20.04     |
| [PyTorch 2.1.2](https://github.com/aws-neuron/deep-learning-containers/blob/2.20.2/docker/pytorch/training/2.1.2/Dockerfile.neuronx)               | torch-neuronx, neuronx_distributed                       | Neuron 2.20.2      | training  | trn1, inf2                   | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:2.1.2-neuronx-py310-sdk2.20.2-ubuntu20.04      |
| [PyTorch 1.13.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.20.2/docker/pytorch/inference/1.13.1/Dockerfile.neuron)             | torch-neuron                                             | Neuron 2.20.2      | inference | inf1                         | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuron:1.13.1-neuron-py310-sdk2.20.2-ubuntu20.04      |
| [PyTorch 1.13.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.20.2/docker/pytorch/inference/1.13.1/Dockerfile.neuronx)            | torch-neuronx, transformers-neuronx, neuronx_distributed | Neuron 2.20.2      | inference | trn1,inf2                    | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.20.2-ubuntu20.04    |
| [PyTorch 1.13.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.20.2/docker/pytorch/training/1.13.1/Dockerfile.neuronx)             | torch-neuronx, neuronx_distributed                       | Neuron 2.20.2      | training  | trn1, inf2                   | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.20.2-ubuntu20.04     |
| [Tensorflow 2.10.1](https://github.com/aws/deep-learning-containers/blob/master/tensorflow/inference/docker/2.10/py3/sdk2.17.0/Dockerfile.neuron)  | tensorflow-neuron                                        | Neuron 2.17.0      | inference | inf1                         | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference-neuron:2.10.1-neuron-py310-sdk2.17.0-ubuntu20.04   |
| [Tensorflow 2.10.1](https://github.com/aws/deep-learning-containers/blob/master/tensorflow/inference/docker/2.10/py3/sdk2.17.0/Dockerfile.neuronx) | tensorflow-neuronx                                       | Neuron 2.17.0      | inference | trn1,inf2                    | 3.10 (py310)           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference-neuronx:2.10.1-neuronx-py310-sdk2.17.0-ubuntu20.04 |

Prior Neuron Containers
=================

| Framework                                                                                                                                        | Neuron Package    | Neuron SDK Version | Job Type  | Supported EC2 Instance Types | Python Version Options | Example URL                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|--------------------|-----------|------------------------------|------------------------|------------------------------------------------------------------------------------------------------------------|
| [Tensorflow 1.15.5](https://github.com/aws/deep-learning-containers/blob/master/tensorflow/inference/docker/1.15/py3/sdk2.8.0/Dockerfile.neuron) | tensorflow-neuron | Neuron 2.8.0       | inference | inf1                         | 3.8 (py38)             | 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference-neuron:1.15.5-neuron-py38-sdk2.8.0-ubuntu20.04 |
| [MXNet 1.8.0](https://github.com/aws/deep-learning-containers/blob/master/mxnet/inference/docker/1.8/py3/sdk2.5.0/Dockerfile.neuron)             | mx_neuron         | Neuron 2.5.0       | inference | inf1                         | 3.8 (py38)             | 763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference-neuron:1.8.0-neuron-py38-sdk2.5.0-ubuntu20.04       |

Prior EC2 Framework Container Versions
==============
| Framework 			                      |Job Type 			   |Horovod Options 			     |CPU/GPU 	  |Python Version Options 			      |Example URL 			                                                                               |
|---------------------------------------------|------------------------|---------------------------------|------------|---------------------------------------|----------------------------------------------------------------------------------------------------|
| PyTorch 2.6.0     |training	|No			|CPU 		| 3.12 (py312)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-cpu-py312-ubuntu22.04-ec2             |
| PyTorch 2.6.0     |training	|No			|GPU 		| 3.12 (py312)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-ec2       |
| PyTorch 2.5.1     |training	|No			|CPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-ec2             |
| PyTorch 2.5.1     |training	|No			|GPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-ec2       |
| PyTorch 2.5.1     |inference	|No			|CPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.5.1-cpu-py311-ubuntu22.04-ec2            |
| PyTorch 2.5.1     |inference	|No			|GPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.5.1-gpu-py311-cu124-ubuntu22.04-ec2     |
| PyTorch 2.4.0     |training	|No			|CPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-cpu-py311-ubuntu22.04-ec2             |
| PyTorch 2.4.0     |training	|No			|GPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-ec2       |
| PyTorch 2.4.0     |inference	|No			|CPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.4.0-cpu-py311-ubuntu22.04-ec2            |
| PyTorch 2.4.0     |inference	|No			|GPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.4.0-gpu-py311-cu124-ubuntu22.04-ec2      |

Prior SageMaker Framework Container Versions
==============
| Framework 			                      |Job Type 			   |Horovod Options 			     |CPU/GPU 	  |Python Version Options 			      |Example URL 			                                                                               |
|---------------------------------------------|------------------------|---------------------------------|------------|---------------------------------------|----------------------------------------------------------------------------------------------------|
| PyTorch 2.6.0     | training	| No			  | CPU 	  | 3.12 (py312)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-cpu-py312-ubuntu22.04-sagemaker           |
| PyTorch 2.6.0     | training	| No			  | GPU 	  | 3.12 (py312)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker     |
| PyTorch 2.5.1     | training	| No			  | CPU 	  | 3.11 (py311)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-cpu-py311-ubuntu22.04-sagemaker           |
| PyTorch 2.5.1     | training	| No			  | GPU 	  | 3.11 (py311)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker     |
| PyTorch 2.5.1     | inference	| No			  | CPU 	  | 3.11 (py311)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.5.1-cpu-py311-ubuntu22.04-sagemaker           |
| PyTorch 2.5.1     | inference	| No			  | GPU 	  | 3.11 (py311)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker     |
| PyTorch 2.4.0     | training	| No			  | CPU 	  | 3.11 (py311)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-cpu-py311-ubuntu22.04-sagemaker           |
| PyTorch 2.4.0     | training	| No			  | GPU 	  | 3.11 (py311)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker     |
| PyTorch 2.4.0     | inference	| No			  | CPU 	  | 3.11 (py311)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.4.0-cpu-py311-ubuntu22.04-sagemaker           |
| PyTorch 2.4.0     | inference	| No			  | GPU 	  | 3.11 (py311)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker     |
| TensorFlow 2.18.0 | training  | No			  | CPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.18.0-cpu-py310-ubuntu22.04-sagemaker		|
| TensorFlow 2.18.0 | training  | No			  | GPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.18.0-gpu-py310-cu125-ubuntu22.04-sagemaker  |
| TensorFlow 2.18.0 | inference | No			  | CPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.18.0-cpu-py310-ubuntu20.04-sagemaker	    |
| TensorFlow 2.18.0 | inference | No			  | GPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.18.0-gpu-py310-cu122-ubuntu20.04-sagemaker |

Prior EC2 Framework ARM64/Graviton Containers
============================
| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	|Example URL																						|
|-------------------|-----------|---------------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
| PyTorch 2.5.1     |inference	|No			|CPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-arm64:2.5.1-cpu-py311-ubuntu22.04-ec2		     |
| PyTorch 2.5.1     |inference	|No			|GPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-arm64:2.5.1-gpu-py311-cu124-ubuntu22.04-ec2		     |
| PyTorch 2.4.0     |inference	|No			|CPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:2.4.0-cpu-py311-ubuntu22.04-ec2		     |
| PyTorch 2.4.0     |inference	|No			|GPU 		| 3.11 (py311)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:2.4.0-gpu-py311-cu124-ubuntu22.04-ec2		     |

Prior SageMaker Framework ARM64/Graviton Containers
============================
| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	|Example URL																						|
|-------------------|-----------|---------------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
| PyTorch 2.5.1     | inference | No			 | CPU 	   | 3.11 (py311)			    | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-arm64:2.5.1-cpu-py311-ubuntu22.04-sagemaker     |
| PyTorch 2.4.0     | inference | No			 | CPU 	   | 3.11 (py311)			    | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:2.4.0-cpu-py311-ubuntu22.04-sagemaker     |

Prior AutoGluon Training Containers
===============================

| Framework       | AutoGluon Version | Job Type | CPU/GPU | Python Version Options | Example URL                                                                                      |
|-----------------|-------------------|----------|---------|------------------------|--------------------------------------------------------------------------------------------------|
| AutoGluon 1.2.0 | 1.2.0              | training | GPU     | 3.11 (py311)             | 763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-training:1.2.0-gpu-py311-cu124-ubuntu22.04 |
| AutoGluon 1.2.0 | 1.2.0              | training | CPU     | 3.11 (py311)             | 763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-training:1.2.0-cpu-py311-ubuntu22.04       |

Prior AutoGluon Inference Containers
===============================

| Framework       | AutoGluon Version | Job Type  | CPU/GPU | Python Version Options | Example URL                                                                                       |
|-----------------|-------------------|-----------|---------|------------------------|---------------------------------------------------------------------------------------------------|
| AutoGluon 1.2.0 | 1.2.0              | inference | GPU     | 3.11 (py311)             | 763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference:1.2.0-gpu-py311-cu124-ubuntu22.04 |
| AutoGluon 1.2.0 | 1.2.0              | inference | CPU     | 3.11 (py311)             | 763104351884.dkr.ecr.us-west-2.amazonaws.com/autogluon-inference:1.2.0-cpu-py311-ubuntu22.04       |

Prior SageMaker Training Compiler Containers
===============================

| Framework                                    |Job Type	|CPU/GPU 	|Python Version Options	|Example URL																						|
|-----------------------------------------------|-----------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|TensorFlow 2.6.3 with HuggingFace transformers 4.17.0 and SageMaker Training Compiler |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-trcomp-training:2.6.3-transformers4.17.0-gpu-py38-cu112-ubuntu20.04 	|
|PyTorch 1.12.0 with SageMaker Training Compiler    |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-trcomp-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker     |

Prior HuggingFace Training Containers
===============================

| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL																						|
|-----------------------------------------------|-----------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|PyTorch 1.10.2 with HuggingFace transformers    |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04      |
