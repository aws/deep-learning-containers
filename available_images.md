# Framework Support Policy

The framework support policy is live on the [DLC](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/support-policy.html) dev guides.

Anaconda shifted to a commercial licensing model for certain users. Actively maintained DLCs have been migrated to the publicly available open-source version of Conda (conda-forge) from the Anaconda channel.

> Warning: If you are actively using Anaconda to install and manage your packages and their dependencies in a DLC that is no longer actively maintained, you are responsible for complying with the governing license from the Anaconda Repository, if you determine that the terms apply to you. Alternatively, you can migrate to one of the currently-supported DLCs listed in the Supported Frameworks table or you can install packages using conda-forge as a source.

# Available Deep Learning Containers Images

The following table lists the Docker image URLs that will be used by Amazon ECS in task definitions. Replace the `<repository-name>` and `<image-tag>` values based on your desired container.

Once you've selected your desired Deep Learning Containers image, continue with one of the following tutorials:

-   To run training and inference on Deep Learning Containers for Amazon EC2 using MXNet, PyTorch, TensorFlow, and TensorFlow 2, see [Amazon EC2 Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2.html)

-   To run training and inference on Deep Learning Containers for Amazon ECS using MXNet, PyTorch, and TensorFlow, see [Amazon ECS tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs.html)

-   Deep Learning Containers for Amazon EKS offer CPU, GPU, and distributed GPU-based training, as well as CPU and GPU-based inference. To run training and inference on Deep Learning Containers for Amazon EKS using MXNet, PyTorch, and TensorFlow, see [Amazon EKS Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-eks.html)

-   For information on security in Deep Learning Containers, see [Security in AWS Deep Learning Containers](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/security.html)

-   For a list of the latest Deep Learning Containers release notes, see [Release Notes for Deep Learning Containers](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-release-notes.html)


Deep Learning Containers Docker Images are available in the following regions:

| Region 					|Code 				|General Container	|Elastic Inference Container|Neuron Container	|Example URL																				|
|---------------------------|-------------------|-------------------|---------------------------|-------------------|-------------------------------------------------------------------------------------------|
|US East (N. Virginia)		|us-east-1			|Available 			|Available 			        |Available			|763104351884.dkr.ecr.us-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|US East (Ohio)				|us-east-2			|Available 			|Available 			        |Available			|763104351884.dkr.ecr.us-east-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|US West (N. California)	|us-west-1			|Available 			|None 			            |None				|763104351884.dkr.ecr.us-west-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|US West (Oregon)			|us-west-2			|Available 			|Available 			        |Available			|763104351884.dkr.ecr.us-west-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|Africa (Cape Town)			|af-south-1			|Available			|Available					|None				|626614931356.dkr.ecr.af-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|Asia Pacific (Hong Kong)	|ap-east-1			|Available 			|None 			            |None				|871362719292.dkr.ecr.ap-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|Asia Pacific (Mumbai)		|ap-south-1			|Available 			|None 			            |Available			|763104351884.dkr.ecr.ap-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|Asia Pacific (Osaka)		|ap-northeast-3		|Available 			|None     			        |None				|364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/&lt;repository-name>:&lt;image-tag>		|
|Asia Pacific (Seoul)		|ap-northeast-2		|Available 			|Available 			        |None				|763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>		|
|Asia Pacific (Singapore)	|ap-southeast-1		|Available 			|None 			            |Available			|763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		|
|Asia Pacific (Sydney)		|ap-southeast-2 	|Available 			|None 			            |Available			|763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>		|
|Asia Pacific (Jakarta)		|ap-southeast-3 	|Available 			|None 			            |None			|907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/&lt;repository-name>:&lt;image-tag>		|
|Asia Pacific (Tokyo)		|ap-northeast-1		|Available 			|Available 			        |Available			|763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		|
|Canada (Central)			|ca-central-1		|Available 			|None 			            |None				|763104351884.dkr.ecr.ca-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		|
|EU (Frankfurt) 			|eu-central-1		|Available 			|None 			            |Available			|763104351884.dkr.ecr.eu-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		|
|EU (Ireland) 				|eu-west-1			|Available 			|Available 			        |Available			|763104351884.dkr.ecr.eu-west-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|EU (London) 				|eu-west-2			|Available 			|None 			            |None				|763104351884.dkr.ecr.eu-west-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|EU (Milan)					|eu-south-1			|Available			|None						|None				|692866216735.dkr.ecr.eu-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|EU (Paris) 				|eu-west-3			|Available 			|None 			            |Available			|763104351884.dkr.ecr.eu-west-3.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|EU (Stockholm) 			|eu-north-1			|Available 			|None 			            |None				|763104351884.dkr.ecr.eu-north-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|Middle East (Bahrain) 		|me-south-1			|Available 			|None 			            |None				|217643126080.dkr.ecr.me-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|Middle East (UAE)		|me-central-1 	|Available 			|None 			            |None			|914824155844.dkr.ecr.me-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		|
|South America (Sao Paulo)	|sa-east-1			|Available 			|None 			            |Available			|763104351884.dkr.ecr.sa-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			|
|China (Beijing)			|cn-north-1			|Available 			|None 			            |None				|727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/&lt;repository-name>:&lt;image-tag>		|
|China (Ningxia)			|cn-northwest-1		|Available 			|None 			            |None				|727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/&lt;repository-name>:&lt;image-tag>	|


Note: **eu-north-1** is available starting in the version 2.0 release.
As a result, MXNet-1.4.0 is not available in this region.

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
job type is either ``training`` or ``inference``. Your Python version is
either ``py27``, ``py36``, ``py37``, or ``py38``, depending on availability. Plug this information into the replaceable portions of the URL as shown in the example URL.

You can pin your version by adding the version tag to your URL as follows:

     763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.4.1-cpu-py37-ubuntu18.04-v1.0

EC2 Framework Containers (Tested on EC2, ECS, and EKS only)
============================

| Framework        |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	| Example URL																						                                                                      |
|------------------|-----------|---------------|-----------|-----------------------|--------------------------------------------------------------------------------------------------------|
| PyTorch 1.12.1   |training	|No			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-cpu-py38-ubuntu20.04-ec2              |
| PyTorch 1.12.1   |training	|Yes		|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu116-ubuntu20.04-ec2        |
| PyTorch 1.12.1   |inference	|No			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-cpu-py38-ubuntu20.04-ec2             |
| PyTorch 1.12.1   |inference	|No			|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-gpu-py38-cu116-ubuntu20.04-ec2       |
| TensorFlow 2.10.0 |training	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.10.0-cpu-py39-ubuntu20.04-ec2		       |
| TensorFlow 2.10.0 |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.10.0-gpu-py39-cu112-ubuntu20.04-ec2	  |
| TensorFlow 2.10.0 |inference	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.0-cpu-py39-ubuntu20.04-ec2		      |
| TensorFlow 2.10.0 |inference	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.0-gpu-py39-cu112-ubuntu20.04-ec2	 |
| MXNet 1.9.0      |training	|Yes			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38-ubuntu20.04-ec2		            |
| MXNet 1.9.0      |training	|Yes			|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38-cu112-ubuntu20.04-ec2	       |
| MXNet 1.9.0      |inference	|No			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38-ubuntu20.04-ec2             |
| MXNet 1.9.0      |inference	|No			|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38-cu112-ubuntu20.04-ec2	      |


SageMaker Framework Containers (SM support only)
============================

| Framework         | Job Type  | Horovod Options | CPU/GPU   | Python Version Options | Example URL																						           |
|-------------------|-----------|-----------------|-----------|------------------------|---------------------------------------------------------------------------------------------------------------|
| PyTorch 1.12.1    | training  | No			  | CPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-cpu-py38-ubuntu20.04-sagemaker           |
| PyTorch 1.12.1    | training  | Yes			  | GPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker     |
| PyTorch 1.12.1    | inference | No			  | CPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-cpu-py38-ubuntu20.04-sagemaker          |
| PyTorch 1.12.1    | inference | No			  | GPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker    |
| MXNet 1.9.0       | training  | Yes			  | CPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38-ubuntu20.04-sagemaker		       |
| MXNet 1.9.0       | training  | Yes			  | GPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38-cu112-ubuntu20.04-sagemaker	       |
| MXNet 1.9.0       | inference | No			  | CPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38-ubuntu20.04-sagemaker             |
| MXNet 1.9.0       | inference | No			  | GPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38-cu112-ubuntu20.04-sagemaker	   |
| TensorFlow 2.10.0 | training  | Yes			  | CPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.10.0-cpu-py39-ubuntu20.04-sagemaker		   |
| TensorFlow 2.10.0 | training  | Yes			  | GPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.10.0-gpu-py39-cu112-ubuntu20.04-sagemaker  |
| TensorFlow 2.10.0 | inference | Yes			  | CPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.0-cpu-py39-ubuntu20.04-sagemaker	   |
| TensorFlow 2.10.0 | inference | Yes			  | GPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.0-gpu-py39-cu112-ubuntu20.04-sagemaker |


EC2 Framework Graviton Containers (EC2, ECS, and EKS support only)
============================

| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	|Example URL																						|
|-------------------|-----------|---------------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|PyTorch 1.12.1   |inference	|No			|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:1.12.1-cpu-py38-ubuntu20.04-ec2		|
|PyTorch 1.10.0   |inference	|No			|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:1.10.0-cpu-py38-ubuntu20.04-ec2		|
|TensorFlow 2.9.1   |inference	|No			|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-graviton:2.9.1-cpu-py38-ubuntu20.04-ec2		|
|TensorFlow 2.7.0   |inference	|No			|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-graviton:2.7.0-cpu-py38-ubuntu20.04-ec2		|


SageMaker Framework Graviton Containers (SM support only)
============================

| Framework         | Job Type	| Horovod Options| CPU/GPU | Python Version Options	| Example URL																						              |
|-------------------|-----------|----------------|---------|------------------------|-----------------------------------------------------------------------------------------------------------------|
| PyTorch 1.12.1    | inference | No			 | CPU 	   | 3.8 (py38)			    | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:1.12.1-cpu-py38-ubuntu20.04-sagemaker   |
| TensorFlow 2.9.1  | inference | no			 | CPU 	   | 3.9 (py39)			    | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-graviton:2.9.1-cpu-py38-ubuntu20.04-sagemaker |


NVIDIA Triton Inference Containers (SM support only)
============================

| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	|Example URL																						|
|-------------------|-----------|---------------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|NVIDIA Triton Inference Server 22.07    |inference	|No			|GPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:22.07-py3		|
|NVIDIA Triton Inference Server 22.07    |inference	|No			|CPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:22.07-py3-cpu		|
|NVIDIA Triton Inference Server 22.05    |inference	|No			|GPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:22.05-py3		|
|NVIDIA Triton Inference Server 22.05    |inference	|No			|CPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:22.05-py3-cpu		|
|NVIDIA Triton Inference Server 21.08    |inference	|No			|GPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:21.08-py3		|
|NVIDIA Triton Inference Server 21.08    |inference	|No			|CPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:21.08-py3-cpu		|

Habana Training Containers
===============================

| Framework         |Job Type	|Device Type 	|Python Version Options	| SynapseAI Version |Example URL																								 |
|-------------------|-----------|---------------|-----------------------|-------------------|------------------------------------------------------------------------------------------------------------|
|TensorFlow 2.9.1   |training   |HPU            | 3.8 (py38)            |1.5.0              |763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training-habana:2.9.1-hpu-py38-synapseai1.5.0-ubuntu20.04 |
|PyTorch 1.11.0     |training   |HPU            | 3.8 (py38)            |1.5.0              |763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-habana:1.11.0-hpu-py38-synapseai1.5.0-ubuntu20.04 |
|TensorFlow 2.8.0   |training   |HPU            | 3.8 (py38)            |1.4.1              |763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training-habana:2.8.0-hpu-py38-synapseai1.4.1-ubuntu20.04 |
|PyTorch 1.10.2     |training   |HPU            | 3.8 (py38)            |1.4.1              |763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-habana:1.10.2-hpu-py38-synapseai1.4.1-ubuntu20.04 |


AutoGluon Training Containers
===============================

| Framework       | AutoGluon Version | Job Type | CPU/GPU | Python Version Options | Example URL                                                                                      |
|-----------------|-------------------|----------|---------|------------------------|--------------------------------------------------------------------------------------------------|
| AutoGluon 0.5.2 | 0.5.2             | training | GPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:0.5.2-gpu-py38-cu112-ubuntu20.04 |
| AutoGluon 0.5.2 | 0.5.2             | training | CPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:0.5.2-cpu-py38-ubuntu20.04       |

AutoGluon Inference Containers
===============================

| Framework       | AutoGluon Version | Job Type  | CPU/GPU | Python Version Options | Example URL                                                                                       |
|-----------------|-------------------|-----------|---------|------------------------|---------------------------------------------------------------------------------------------------|
| AutoGluon 0.5.2 | 0.5.2             | inference | GPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-inference:0.5.2-gpu-py38-cu112-ubuntu20.04 |
| AutoGluon 0.5.2 | 0.5.2             | inference | CPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-inference:0.5.2-cpu-py38-ubuntu20.04       |

HuggingFace Training Containers
===============================

| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL																						|
|-----------------------------------------------|-----------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|PyTorch 1.10.2 with HuggingFace transformers    |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04      |
|TensorFlow 2.6.3 with HuggingFace transformers |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-training:2.6.3-transformers4.17.0-gpu-py38-cu112-ubuntu20.04 	|


HuggingFace Inference Containers
===============================

| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL																						|
|-----------------------------------------------|-----------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|PyTorch 1.10.2 with HuggingFace transformers    |inference	|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-cpu-py38-ubuntu20.04		|
|PyTorch 1.10.2 with HuggingFace transformers    |inference	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04		|
|TensorFlow 2.6.3 with HuggingFace transformers |inference	|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.6.3-transformers4.17.0-cpu-py38-ubuntu20.04 	|
|TensorFlow 2.6.3 with HuggingFace transformers |inference	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.6.3-transformers4.17.0-gpu-py38-cu112-ubuntu20.04 	|

HuggingFace Neuron Inference Containers
===============================

|Framework                                  |Job Type   |Python Version Options |Example URL                                                                                                |
|-------------------------------------------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------------|
|PyTorch 1.10.2 with Neuron Inference and HuggingFace transformers |inference 	|3.7 (py37) 	        |763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuron:1.10.2-transformers4.20.1-neuron-py37-sdk1.19.1-ubuntu18.04        |

SageMaker Training Compiler Containers
===============================

| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL																						|
|-----------------------------------------------|-----------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
| TensorFlow 2.10.0                              |training   |GPU        | 3.9 (py39)            | 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.10.0-gpu-py39-cu112-ubuntu20.04-sagemaker     |
|PyTorch 1.11.0 with HuggingFace transformers 4.21.1 and SageMaker Training Compiler    |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-trcomp-training:1.11.0-transformers4.21.1-gpu-py38-cu113-ubuntu20.04      |
|TensorFlow 2.6.3 with HuggingFace transformers 4.17.0 and SageMaker Training Compiler |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-trcomp-training:2.6.3-transformers4.17.0-gpu-py38-cu112-ubuntu20.04 	|


Neuron Containers
=================

|Framework      |Neuron Package |Neuron SDK Version |Job Type   |Instances  |Python Version Options |Example URL                                                                                                |
|---------------|---------------|-------------------|-----------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------------|
|PyTorch 1.11.0 |torch-neuronx  |Neuron 2.4.0       |training 	|trn1   |3.8 (py38) 	        |763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuron:1.11.0-neuron-py38-sdk2.4.0-ubuntu20.04        |
|PyTorch 1.10.2 |torch-neuron   |Neuron 1.19.0      |inference 	|inf1   |3.7 (py37) 	        |763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuron:1.10.2-neuron-py37-sdk1.19.0-ubuntu18.04        |
|MXNet 1.8.0    |mx_neuron      |Neuron 1.18.0      |inference 	|inf1   |3.7 (py37) 	        |763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference-neuron:1.8.0-neuron-py37-sdk1.18.0-ubuntu18.04          |


Elastic Inference Containers
============================

|Framework                                  |Job Type   |Python Version Options |Example URL                                                                                                |
|-------------------------------------------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------------|
| PyTorch 1.5.1 with Elastic Inference        |inference 	|3.8 (py38) 	        | 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-eia:1.5.1-cpu-py38-ubuntu20.04        |


Prior EC2 Framework Container Versions
==============
| Framework 			                      |Job Type 			   |Horovod Options 			     |CPU/GPU 	  |Python Version Options 			      |Example URL 			                                                                               |
|---------------------------------------------|------------------------|---------------------------------|------------|---------------------------------------|----------------------------------------------------------------------------------------------------|
| PyTorch 1.11.0   |training	|No			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-cpu-py38-ubuntu20.04-ec2              |
| PyTorch 1.11.0   |training	|Yes		|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu115-ubuntu20.04-ec2        |
| PyTorch 1.11.0   |inference	|No			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.11.0-cpu-py38-ubuntu20.04-ec2             |
| PyTorch 1.11.0   |inference	|No			|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.11.0-gpu-py38-cu115-ubuntu20.04-ec2       |
| PyTorch 1.10.2   |training	|No			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.2-cpu-py38-ubuntu20.04-ec2              |
| PyTorch 1.10.2   |training	|Yes		|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.2-gpu-py38-cu113-ubuntu20.04-ec2        |
| PyTorch 1.10.2   |inference	|No			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.2-cpu-py38-ubuntu20.04-ec2             |
| PyTorch 1.10.2   |inference	|No			|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.2-gpu-py38-cu113-ubuntu20.04-ec2       |
| TensorFlow 2.7.0   |training	|Yes			|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.7.0-cpu-py38-ubuntu20.04-ec2		|
| TensorFlow 2.7.0   |training	|Yes			|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.7.0-gpu-py38-cu112-ubuntu20.04-ec2	|
| TensorFlow 2.7.0   |inference	|Yes			|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.7.0-cpu-py38-ubuntu20.04-ec2		|
| TensorFlow 2.7.0   |inference	|Yes			|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.7.0-gpu-py38-cu112-ubuntu20.04-ec2	|
| TensorFlow 2.8.0 |training	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.8.0-cpu-py39-ubuntu20.04-ec2		       |
| TensorFlow 2.8.0 |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.8.0-gpu-py39-cu112-ubuntu20.04-ec2	  |
| TensorFlow 2.8.0 |inference	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.8.0-cpu-py39-ubuntu20.04-ec2		      |
| TensorFlow 2.8.0 |inference	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.8.0-gpu-py39-cu112-ubuntu20.04-ec2	 |
| TensorFlow 2.9.1 |training	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9.1-cpu-py39-ubuntu20.04-ec2		       |
| TensorFlow 2.9.1 |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9.1-gpu-py39-cu112-ubuntu20.04-ec2	  |
| TensorFlow 2.9.0 |inference	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.0-cpu-py39-ubuntu20.04-ec2		      |
| TensorFlow 2.9.0 |inference	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.0-gpu-py39-cu112-ubuntu20.04-ec2	 |

Prior SageMaker Framework Container Versions
==============
| Framework 			                      |Job Type 			   |Horovod Options 			     |CPU/GPU 	  |Python Version Options 			      |Example URL 			                                                                               |
|---------------------------------------------|------------------------|---------------------------------|------------|---------------------------------------|----------------------------------------------------------------------------------------------------|
| TensorFlow 2.9.1 |training	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9.1-cpu-py39-ubuntu20.04-sagemaker		       |
| TensorFlow 2.9.1 |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9.1-gpu-py39-cu112-ubuntu20.04-sagemaker	 |
| TensorFlow 2.9.0 |inference	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.0-cpu-py39-ubuntu20.04-sagemaker		      |
| TensorFlow 2.9.0 |inference	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.0-gpu-py39-cu112-ubuntu20.04-sagemaker	 |
| TensorFlow 2.8.0 |training	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.8.0-cpu-py39-ubuntu20.04-sagemaker		       |
| TensorFlow 2.8.0 |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.8.0-gpu-py39-cu112-ubuntu20.04-sagemaker	  |
| TensorFlow 2.8.0 |inference	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.8.0-cpu-py39-ubuntu20.04-sagemaker		      |
| TensorFlow 2.8.0 |inference	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.8.0-gpu-py39-cu112-ubuntu20.04-sagemaker	 |
| TensorFlow 2.7.1 |training	| Yes			          |CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.7.1-cpu-py38-ubuntu20.04-sagemaker		       |
| TensorFlow 2.7.1 |training	| Yes			          |GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.7.1-gpu-py38-cu112-ubuntu20.04-sagemaker	  |
| TensorFlow 2.7.0 |inference	| Yes			          |CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.7.0-cpu-py38-ubuntu20.04-sagemaker		      |
| TensorFlow 2.7.0 |inference	| Yes			          |GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.7.0-gpu-py38-cu112-ubuntu20.04-sagemaker	 |
| PyTorch 1.11.0   |training	| No			           |CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-cpu-py38-ubuntu20.04-sagemaker            |
| PyTorch 1.11.0   |training	| Yes			           |GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker      |
| PyTorch 1.11.0   |inference	| No			           |CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.11.0-cpu-py38-ubuntu20.04-sagemaker           |
| PyTorch 1.11.0   |inference	| No			           |GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker     |
| PyTorch 1.10.2   |training	| No			           |CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.2-cpu-py38-ubuntu20.04-sagemaker		         |
| PyTorch 1.10.2   |training	| Yes			          |GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.10.2-gpu-py38-cu113-ubuntu20.04-sagemaker	    |
| PyTorch 1.10.2   |inference	| No			           |CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.2-cpu-py38-ubuntu20.04-sagemaker		        |
| PyTorch 1.10.2   |inference	| No			           |GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.2-gpu-py38-cu113-ubuntu20.04-sagemaker	   |

Prior General Framework Container Versions
==============

| Framework 			                      |Job Type 			   |Horovod Options 			     |CPU/GPU 	  |Python Version Options 			      |Example URL 			                                                                               |
|---------------------------------------------|------------------------|---------------------------------|------------|---------------------------------------|----------------------------------------------------------------------------------------------------|
|TensorFlow 2.6.2   |training	|Yes			|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.6.2-cpu-py38-ubuntu20.04		|
|TensorFlow 2.6.2   |training	|Yes			|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.6.2-gpu-py38-cu112-ubuntu20.04	|
|TensorFlow 2.6.0   |inference	|Yes			|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.6.0-cpu-py38-ubuntu20.04		|
|TensorFlow 2.6.0   |inference	|Yes			|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.6.0-gpu-py38-cu112-ubuntu20.04	|


Prior AutoGluon Training Containers
===============================

| Framework       | AutoGluon Version | Job Type | CPU/GPU | Python Version Options | Example URL                                                                                      |
|-----------------|-------------------|----------|---------|------------------------|--------------------------------------------------------------------------------------------------|
| AutoGluon 0.4.3 | 0.4.3             | training | GPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:0.4.3-gpu-py38-cu112-ubuntu20.04 |
| AutoGluon 0.4.3 | 0.4.3             | training | CPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:0.4.3-cpu-py38-ubuntu20.04       |

Prior AutoGluon Inference Containers
===============================

| Framework       | AutoGluon Version | Job Type  | CPU/GPU | Python Version Options | Example URL                                                                                       |
|-----------------|-------------------|-----------|---------|------------------------|---------------------------------------------------------------------------------------------------|
| AutoGluon 0.4.3 | 0.4.3             | inference | GPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-inference:0.4.3-gpu-py38-cu112-ubuntu20.04 |
| AutoGluon 0.4.3 | 0.4.3             | inference | CPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-inference:0.4.3-cpu-py38-ubuntu20.04       |


Prior Neuron Inference Container Versions
==============
| Framework 			                    |Job Type 	 |Python Version Options    |Example URL 			                                                                                    |
|-------------------------------------------|------------|--------------------------|-----------------------------------------------------------------------------------------------------------|
|PyTorch 1.10.1 with Neuron Inference        |inference 	|3.7 (py37) 	        |763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-neuron:1.10.1-neuron-py37-sdk1.17.1-ubuntu18.04        |


Prior SageMaker Training Compiler Containers
===============================
| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL																						|
|-----------------------------------------------|-----------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|PyTorch 1.10.2 with HuggingFace transformers 4.17.0 and SageMaker Training Compiler    |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-trcomp-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04      |
