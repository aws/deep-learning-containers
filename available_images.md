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

| Region 					               | Code 				        |General Container	|Neuron Container	| Example URL																				                                                           |
|----------------------------|------------------|-------------------|-------------------|-------------------------------------------------------------------------------------------|
| US East (N. Virginia)		    | us-east-1			     |Available 			|Available			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| US East (Ohio)				         | us-east-2			     |Available 			|Available			| 763104351884.dkr.ecr.us-east-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| US West (N. California)	   | us-west-1			     |Available 			|None				| 763104351884.dkr.ecr.us-west-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| US West (Oregon)			        | us-west-2			     |Available 			|Available			| 763104351884.dkr.ecr.us-west-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| Africa (Cape Town)			      | af-south-1			    |Available			|None				| 626614931356.dkr.ecr.af-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| Asia Pacific (Hong Kong)	  | ap-east-1			     |Available 			|None				| 871362719292.dkr.ecr.ap-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| Asia Pacific (Mumbai)		    | ap-south-1			    |Available 			|Available			| 763104351884.dkr.ecr.ap-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| Asia Pacific (Hyderabad)		 | ap-south-2			    |Available 			|None			| 772153158452.dkr.ecr.ap-south-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| Asia Pacific (Osaka)		     | ap-northeast-3		 |Available 			|None				| 364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Seoul)		     | ap-northeast-2		 |Available 			|None				| 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Singapore)	  | ap-southeast-1		 |Available 			|Available			| 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Sydney)		    | ap-southeast-2 	 |Available 			|Available			| 763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Jakarta)		   | ap-southeast-3 	 |Available 			|None			| 907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Melbourne)		 | ap-southeast-4 	 |Available 			|None			| 457447274322.dkr.ecr.ap-southeast-4.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Asia Pacific (Tokyo)		     | ap-northeast-1		 |Available 			|Available			| 763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		   |
| Canada (Central)			        | ca-central-1		   |Available 			|None				| 763104351884.dkr.ecr.ca-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		     |
| EU (Frankfurt) 			         | eu-central-1		   |Available 			|Available			| 763104351884.dkr.ecr.eu-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		     |
| EU (Zurich) 			            | eu-central-2		   |Available 			|None			| 380420809688.dkr.ecr.eu-central-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>		     |
| EU (Ireland) 				          | eu-west-1			     |Available 			|Available			| 763104351884.dkr.ecr.eu-west-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| EU (London) 				           | eu-west-2			     |Available 			|None				| 763104351884.dkr.ecr.eu-west-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| EU (Milan)					            | eu-south-1			    |Available			|None				| 692866216735.dkr.ecr.eu-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| EU (Spain)					            | eu-south-2			    |Available			|None				| 503227376785.dkr.ecr.eu-south-2.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| EU (Paris) 				            | eu-west-3			     |Available 			|Available			| 763104351884.dkr.ecr.eu-west-3.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| EU (Stockholm) 			         | eu-north-1			    |Available 			|None				| 763104351884.dkr.ecr.eu-north-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| Israel (Tel Aviv) 			      | il-central-1			  |Available 			|None				| 763104351884.dkr.ecr.il-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			    |
| Middle East (Bahrain) 		   | me-south-1			    |Available 			|None				| 217643126080.dkr.ecr.me-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			      |
| Middle East (UAE)		        | me-central-1 	   |Available 			|None			| 914824155844.dkr.ecr.me-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>		     |
| South America (Sao Paulo)	 | sa-east-1			     |Available 			|Available			| 763104351884.dkr.ecr.sa-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag>			       |
| China (Beijing)			         | cn-north-1			    |Available 			|None				| 727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/&lt;repository-name>:&lt;image-tag>		    |
| China (Ningxia)			         | cn-northwest-1		 |Available 			|None				| 727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/&lt;repository-name>:&lt;image-tag>	 |


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
| PyTorch 2.0.1   |training	|Yes			|CPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-cpu-py310-ubuntu20.04-ec2             |
| PyTorch 2.0.1   |training	|Yes			|GPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2       |      |
| PyTorch 2.0.1   |inference	|No			|CPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310-ubuntu20.04-ec2             |
| PyTorch 2.0.1   |inference	|No			|GPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2       |
| TensorFlow 2.12.0 |training	|Yes			|CPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.12.0-cpu-py310-ubuntu20.04-ec2		       |
| TensorFlow 2.12.0 |training	|Yes			|GPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.12.0-gpu-py310-cu118-ubuntu20.04-ec2	  |
| TensorFlow 2.12.1 |inference	|No			|CPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.12.1-cpu-py310-ubuntu20.04-ec2		       |
| TensorFlow 2.12.1 |inference	|No			|GPU 		| 3.10 (py310)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.12.1-gpu-py310-cu118-ubuntu20.04-ec2	  |
| MXNet 1.9.0      |training	|Yes			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38-ubuntu20.04-ec2		            |
| MXNet 1.9.0      |training	|Yes			|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38-cu112-ubuntu20.04-ec2	       |
| MXNet 1.9.0      |inference	|No			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38-ubuntu20.04-ec2             |
| MXNet 1.9.0      |inference	|No			|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38-cu112-ubuntu20.04-ec2	      |


SageMaker Framework Containers (SM support only)
============================

| Framework         | Job Type  | Horovod Options | CPU/GPU   | Python Version Options | Example URL																						           |
|-------------------|-----------|-----------------|-----------|------------------------|---------------------------------------------------------------------------------------------------------------|
| PyTorch 2.0.1    | inference	| No			  | CPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-cpu-py310-ubuntu20.04-sagemaker          |
| PyTorch 2.0.1    | inference	| No			  | GPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker    |
| PyTorch 2.0.0    | training   | No 			  | CPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-cpu-py310-ubuntu20.04-sagemaker           |
| PyTorch 2.0.0    | training   | Yes			  | GPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310-cu118-ubuntu20.04-sagemaker     |
| TensorFlow 2.12.0 | training  | Yes			  | CPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.12.0-cpu-py310-ubuntu20.04-sagemaker		   |
| TensorFlow 2.12.0 | training  | Yes			  | GPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.12.0-gpu-py310-cu118-ubuntu20.04-sagemaker  |
| TensorFlow 2.12.1 | inference | No			  | CPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.12.1-cpu-py310-ubuntu20.04-sagemaker	   |
| TensorFlow 2.12.1 | inference | No			  | GPU 	  | 3.10 (py310)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.12.1-gpu-py310-cu118-ubuntu20.04-sagemaker |
| MXNet 1.9.0       | training  | Yes			  | CPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38-ubuntu20.04-sagemaker		       |
| MXNet 1.9.0       | training  | Yes			  | GPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-gpu-py38-cu112-ubuntu20.04-sagemaker	       |
| MXNet 1.9.0       | inference | No			  | CPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.9.0-cpu-py38-ubuntu20.04-sagemaker             |
| MXNet 1.9.0       | inference | No			  | GPU 	  | 3.8 (py38)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.9.0-gpu-py38-cu112-ubuntu20.04-sagemaker	   |


EC2 Framework Graviton Containers (EC2, ECS, and EKS support only)
============================

| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	|Example URL																						|
|-------------------|-----------|---------------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|PyTorch 2.0.1   |inference	|No			|CPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-ec2		|
|TensorFlow 2.12.1   |inference	|No			|CPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-graviton:2.12.1-cpu-py310-ubuntu20.04-ec2		|

SageMaker Framework Graviton Containers (SM support only)
============================

| Framework         | Job Type	| Horovod Options| CPU/GPU | Python Version Options	| Example URL																						              |
|-------------------|-----------|----------------|---------|------------------------|-----------------------------------------------------------------------------------------------------------------|
| PyTorch 2.0.1    | inference | No			 | CPU 	   | 3.10 (py310)			    | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:2.0.1-cpu-py310-ubuntu20.04-sagemaker   |
| TensorFlow 2.12.1  | inference | No			 | CPU 	   | 3.10 (py310)			    | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-graviton:2.12.1-cpu-py310-ubuntu20.04-sagemaker |

NVIDIA Triton Inference Containers (SM support only)
============================
**Note**: The following versions of the 23.`<XY>` container are supported: `23.01, 23.02, 23.03`
| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	|Example URL																						|
|-------------------|-----------|---------------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|NVIDIA Triton Inference Server 23.`<XY>`    |inference	|No			|GPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:23.`<XY>`-py3		|

**Note**: The following versions of the 22.`<XY>` container are supported:
`22.05, 22.07, 22.08, 22.09, 22.10, 22.12`

| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	|Example URL																						|
|-------------------|-----------|---------------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|NVIDIA Triton Inference Server 22.`<XY>`    |inference	|No			|GPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:22.`<XY>`-py3		|
|NVIDIA Triton Inference Server 22.`<XY>`    |inference	|No			|CPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:22.`<XY>`-py3-cpu		|
|NVIDIA Triton Inference Server 21.08    |inference	|No			|GPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:21.08-py3		|
|NVIDIA Triton Inference Server 21.08    |inference	|No			|CPU 		| 3.8 (py38)			|007439368137.dkr.ecr.us-east-2.amazonaws.com/sagemaker-tritonserver:21.08-py3-cpu		|


Large Model Inference Containers
===============================
| Framework                                                                                                                   |Job Type	| Accelerator 	 | Python Version Options	 | Example URL																						                                                              |
|-----------------------------------------------------------------------------------------------------------------------------|-----------|---------------|-------------------------|------------------------------------------------------------------------------------------------|
| DJLServing 0.22.1 with FasterTransformer 5.3.0, HuggingFace Transformers 4.27.3, and HuggingFace Accelerate 0.17.1          |inference	| GPU 		        | 3.9 (py39)			           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.22.1-fastertransformer5.3.0-cu118 |
| DJLServing 0.22.1 with DeepSpeed 0.9.2, HuggingFace Transformers 4.29.2, Diffusers 0.15.0 and HuggingFace Accelerate 0.19.0 |inference	| GPU 		        | 3.9 (py39)			           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.22.1-deepspeed0.9.2-cu118		       |
| DJLServing 0.22.1 with Neuron SDK 2.10.0, TransformersNeuronX 0.3.0 and HuggingFace Transformers 4.28.1                     |inference	| Neuron 		     | 3.8 (py38)			           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.22.1-neuronx-sdk2.10.0            |
| DJLServing 0.21.0 with FasterTransformer 5.3.0, HuggingFace Transformers 4.25.1, and HuggingFace Accelerate 0.15.0          |inference	| GPU 		        | 3.9 (py39)			           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.21.0-fastertransformer5.3.0-cu117 |
| DJLServing 0.21.0 with DeepSpeed 0.8.3, HuggingFace Transformers 4.26.0, and HuggingFace Accelerate 0.16.0                  |inference	| GPU 		        | 3.9 (py39)			           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.3-cu117		       |
| DJLServing 0.20.0 with DeepSpeed 0.7.5, HuggingFace Transformers 4.23.1, and HuggingFace Accelerate 0.13.2                  |inference	| GPU 		        | 3.8 (py38)			           | 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.20.0-deepspeed0.7.5-cu116		       |


DJL CPU Full Inference Containers
===============================
| Framework         |Job Type	|CPU/GPU 	|Python Version Options	| Example URL																						                                          |
|-------------------|-----------|-----------|-----------------------|----------------------------------------------------------------------------|
| DJLServing 0.22.1 |inference	|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.22.1-cpu-full |
| DJLServing 0.21.0 |inference	|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.21.0-cpu-full |

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

| Framework       | AutoGluon Version  | Job Type | CPU/GPU | Python Version Options | Example URL                                                                                      |
|-----------------|--------------------|----------|---------|------------------------|--------------------------------------------------------------------------------------------------|
| AutoGluon 0.7.0 | 0.7.0              | training | GPU     | 3.9 (py39)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:0.7.0-gpu-py39-cu117-ubuntu20.04 |
| AutoGluon 0.7.0 | 0.7.0              | training | CPU     | 3.9 (py39)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:0.7.0-cpu-py39-ubuntu20.04       |

AutoGluon Inference Containers
===============================

| Framework       | AutoGluon Version  | Job Type  | CPU/GPU | Python Version Options | Example URL                                                                                       |
|-----------------|--------------------|-----------|---------|------------------------|---------------------------------------------------------------------------------------------------|
| AutoGluon 0.7.0 | 0.7.0              | inference | GPU     | 3.9 (py39)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-inference:0.7.0-gpu-py39-cu117-ubuntu20.04 |
| AutoGluon 0.7.0 | 0.7.0              | inference | CPU     | 3.9 (py39)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-inference:0.7.0-cpu-py39-ubuntu20.04       |

HuggingFace Training Containers
===============================

| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL                                                                                                                        |
|-----------------------------------------------|-----------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------|
|PyTorch 2.0.0 with HuggingFace transformers    |training	|GPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04     |
|PyTorch 1.13.1 with HuggingFace transformers   |training	|GPU 		| 3.9 (py39)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04     |
|TensorFlow 2.6.3 with HuggingFace transformers |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-training:2.6.3-transformers4.17.0-gpu-py38-cu112-ubuntu20.04 	|


HuggingFace Inference Containers
===============================

| Framework                                        |Job Type	|CPU/GPU 	|Python Version Options	|Example URL                                                                                                                        |
|--------------------------------------------------|------------|-----------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------|
|PyTorch 2.0.0 with HuggingFace transformers       |inference	|CPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-cpu-py310-ubuntu20.04		    |
|PyTorch 2.0.0 with HuggingFace transformers       |inference	|GPU 		| 3.10 (py310)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04    |
|PyTorch 1.13.1 with HuggingFace transformers      |inference	|CPU 		| 3.9 (py39)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04		    |
|PyTorch 1.13.1 with HuggingFace transformers      |inference	|GPU 		| 3.9 (py39)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04    |
|TensorFlow 2.11.1 with HuggingFace transformers   |inference	|CPU 		| 3.9 (py39)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.11.1-transformers4.26.0-cpu-py39-ubuntu20.04		|
|TensorFlow 2.11.1 with HuggingFace transformers   |inference	|GPU 		| 3.9 (py39)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-inference:2.11.1-transformers4.26.0-gpu-py39-cu112-ubuntu20.04	|

HuggingFace Text Generation Inference Containers
===============================

| Framework                          | Job Type   | CPU/GPU | Python Version Options | Example URL                                                                                                                |
|------------------------------------|------------|---------|------------------------|----------------------------------------------------------------------------------------------------------------------------|
| PyTorch 2.0.0 with HuggingFace TGI | inference  | GPU     | 3.9 (py39)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.0.0-tgi0.8.2-gpu-py39-cu118-ubuntu20.04   |

HuggingFace Neuron/NeuronX Inference Containers
===============================

|Framework                                                         |Neuron SDK Version |Job Type   |Supported EC2 Instance Type |Python Version Options |Example URL                                                                                                                                   |
|------------------------------------------------------------------|-------------------|-----------|----------------------------|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
|PyTorch 1.10.2 with Neuron Inference and HuggingFace transformers |Neuron 1.19.1      |inference  |inf1                        |3.7 (py37)             |763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuron:1.10.2-transformers4.20.1-neuron-py37-sdk1.19.1-ubuntu18.04 |
|PyTorch 1.13.0 with NeuronX Inference and HuggingFace transformers |NeuronX 2.9.1      |inference  |inf2/trn1                        |3.8 (py38)             |763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuronx:1.13.0-transformers4.28.1-neuronx-py38-sdk2.9.1-ubuntu20.04 |

HuggingFace Neuronx Training Containers
===============================
|Framework                                                         |Neuron SDK Version |Job Type   |Supported EC2 Instance Type |Python Version Options |Example URL                                                                                                                                   |
|------------------------------------------------------------------|-------------------|-----------|----------------------------|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
|PyTorch 1.13.0 with Neuronx Training and HuggingFace transformers |Neuronx 2.9.1      |training  |trn1                        |3.8 (py38)             |763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-neuronx:1.13.0-transformers4.28.1-neuronx-py38-sdk2.9.1-ubuntu20.04 |

SageMaker Training Compiler Containers
===============================

| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL																						|
|-----------------------------------------------|-----------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
| TensorFlow 2.10.0                              |training   |GPU        | 3.9 (py39)            | 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.10.0-gpu-py39-cu112-ubuntu20.04-sagemaker     |
|PyTorch 1.13.1 with SageMaker Training Compiler    |training	|GPU 		| 3.9 (py39 )			|763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-trcomp-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker     |
|PyTorch 1.11.0 with HuggingFace transformers 4.21.1 and SageMaker Training Compiler    |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-trcomp-training:1.11.0-transformers4.21.1-gpu-py38-cu113-ubuntu20.04      |


Neuron Containers
=================

|Framework          |Neuron Package     |Neuron SDK Version |Job Type   |Supported EC2 Instance Types |Python Version Options |Example URL                                                                                                          |
|-------------------|-------------------|-------------------|-----------|-----------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------|
|PyTorch 1.13.1     |torch-neuron       |Neuron 2.10.0      |inference  |inf1                         |3.8 (py38)             |763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuron:1.13.1-neuron-py38-sdk2.10.0-ubuntu20.04       |
|PyTorch 1.13.1     |torch-neuronx      |Neuron 2.10.0      |inference  |trn1,inf2                    |3.8 (py38)             |763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:1.13.1-neuronx-py38-sdk2.10.0-ubuntu20.04     |
|Tensorflow 2.10.1  |tensorflow-neuron  |Neuron 2.10.0      |inference  |inf1                         |3.8 (py38)             |763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference-neuron:2.10.1-neuron-py38-sdk2.10.0-ubuntu20.04    |
|Tensorflow 2.10.1  |tensorflow-neuronx |Neuron 2.10.0      |inference  |trn1,inf2                    |3.8 (py38)             |763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference-neuronx:2.10.1-neuronx-py38-sdk2.10.0-ubuntu20.04  |
|Tensorflow 1.15.5  |tensorflow-neuron  |Neuron 2.8.0       |inference  |inf1                         |3.8 (py38)             |763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference-neuron:1.15.5-neuron-py38-sdk2.8.0-ubuntu20.04     |
|MXNet 1.8.0        |mx_neuron          |Neuron 2.5.0       |inference  |inf1                         |3.8 (py38)             |763104351884.dkr.ecr.us-west-2.amazonaws.com/mxnet-inference-neuron:1.8.0-neuron-py38-sdk2.5.0-ubuntu20.04           |
|PyTorch 1.13.1     |torch-neuronx      |Neuron 2.10.0      |training   |trn1, inf2                   |3.8 (py38)             |763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py38-sdk2.10.0-ubuntu20.04      |

Prior EC2 Framework Container Versions
==============
| Framework 			                      |Job Type 			   |Horovod Options 			     |CPU/GPU 	  |Python Version Options 			      |Example URL 			                                                                               |
|---------------------------------------------|------------------------|---------------------------------|------------|---------------------------------------|----------------------------------------------------------------------------------------------------|
| PyTorch 1.13.1   |inference	|No			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.13.1-cpu-py39-ubuntu20.04-ec2             |
| PyTorch 1.13.1   |inference	|No			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.13.1-gpu-py39-cu117-ubuntu20.04-ec2       |
| PyTorch 1.13.1   |training	|No			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-cpu-py39-ubuntu20.04-ec2             |
| PyTorch 1.13.1   |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-ec2       |
| PyTorch 1.12.1   |training	|Yes			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-cpu-py38-ubuntu20.04-ec2             |
| PyTorch 1.12.1   |training	|Yes		    |GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu116-ubuntu20.04-ec2       |
| PyTorch 1.12.1   |inference	|No			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-cpu-py38-ubuntu20.04-ec2             |
| PyTorch 1.12.1   |inference	|No			|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-gpu-py38-cu116-ubuntu20.04-ec2       |
| PyTorch 1.11.0   |training	|Yes			|CPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-cpu-py38-ubuntu20.04-ec2              |
| PyTorch 1.11.0   |training	|Yes		|GPU 		| 3.8 (py38)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu115-ubuntu20.04-ec2        |
| TensorFlow 2.11.1 |inference	|No			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.11.1-cpu-py39-ubuntu20.04-ec2		      |
| TensorFlow 2.11.1 |inference	|No			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.11.1-gpu-py39-cu112-ubuntu20.04-ec2	 |
| TensorFlow 2.11.0 |training	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.11.0-cpu-py39-ubuntu20.04-ec2		       |
| TensorFlow 2.11.0 |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.11.0-gpu-py39-cu112-ubuntu20.04-ec2	  |
| TensorFlow 2.11.0 |inference	|No			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.11.0-cpu-py39-ubuntu20.04-ec2		      |
| TensorFlow 2.11.0 |inference	|No			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.11.0-gpu-py39-cu112-ubuntu20.04-ec2	 |
| TensorFlow 2.10.1 |inference	|No			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.1-cpu-py39-ubuntu20.04-ec2		      |
| TensorFlow 2.10.1 |inference	|No			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.1-gpu-py39-cu112-ubuntu20.04-ec2	 |
| TensorFlow 2.10.0 |training	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.10.0-cpu-py39-ubuntu20.04-ec2		       |
| TensorFlow 2.10.0 |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.10.0-gpu-py39-cu112-ubuntu20.04-ec2	  |
| TensorFlow 2.10.0 |inference	|No			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.0-cpu-py39-ubuntu20.04-ec2		      |
| TensorFlow 2.10.0 |inference	|No			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.0-gpu-py39-cu112-ubuntu20.04-ec2	 |
| TensorFlow 2.9.3 |inference	|No			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.3-cpu-py39-ubuntu20.04-ec2		      |
| TensorFlow 2.9.3 |inference	|No			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.3-gpu-py39-cu112-ubuntu20.04-ec2	 |
| TensorFlow 2.9.2 |training	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9.2-cpu-py39-ubuntu20.04-ec2		       |
| TensorFlow 2.9.2 |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9.2-gpu-py39-cu112-ubuntu20.04-ec2	  |
| TensorFlow 2.9.2 |inference	|No			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.2-cpu-py39-ubuntu20.04-ec2		      |
| TensorFlow 2.9.2 |inference	|No			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.2-gpu-py39-cu112-ubuntu20.04-ec2	 |
| TensorFlow 2.8.3 |training	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.8.3-cpu-py39-ubuntu20.04-ec2		       |
| TensorFlow 2.8.3 |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.8.3-gpu-py39-cu112-ubuntu20.04-ec2	  |

Prior SageMaker Framework Container Versions
==============
| Framework 			                      |Job Type 			   |Horovod Options 			     |CPU/GPU 	  |Python Version Options 			      |Example URL 			                                                                               |
|---------------------------------------------|------------------------|---------------------------------|------------|---------------------------------------|----------------------------------------------------------------------------------------------------|
| PyTorch 1.13.1    | inference	| No			  | CPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.13.1-cpu-py39-ubuntu20.04-sagemaker          |
| PyTorch 1.13.1    | inference	| No			  | GPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker    |
| PyTorch 1.13.1    | training	| No			  | CPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-cpu-py39-ubuntu20.04-sagemaker          |
| PyTorch 1.13.1    | training	| Yes			  | GPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker    |
| PyTorch 1.12.1   |training	| Yes			           |CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-cpu-py38-ubuntu20.04-sagemaker            |
| PyTorch 1.12.1   |training	| Yes			           |GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker      |
| PyTorch 1.12.1   |inference	| No			           |CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-cpu-py38-ubuntu20.04-sagemaker           |
| PyTorch 1.12.1   |inference	| No			           |GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker     |
| PyTorch 1.11.0   |training	| Yes			           |CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-cpu-py38-ubuntu20.04-sagemaker            |
| PyTorch 1.11.0   |training	| Yes			           |GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker      |
| TensorFlow 2.11.0 | training  | Yes			  | CPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.11.0-cpu-py39-ubuntu20.04-sagemaker		   |
| TensorFlow 2.11.0 | training  | Yes			  | GPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.11.0-gpu-py39-cu112-ubuntu20.04-sagemaker  |
| TensorFlow 2.11.1 | inference | No			  | CPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.11.1-cpu-py39-ubuntu20.04-sagemaker	   |
| TensorFlow 2.11.1 | inference | No			  | GPU 	  | 3.9 (py39)			   | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.11.1-gpu-py39-cu112-ubuntu20.04-sagemaker |
| TensorFlow 2.10.1 |training	| Yes			|CPU 		| 3.9 (py39)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.10.1-cpu-py39-ubuntu20.04-sagemaker			      |
| TensorFlow 2.10.1 |training	| Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.10.1-gpu-py39-ubuntu20.04-sagemaker		 |
| TensorFlow 2.10.1 |inference	|No			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.1-cpu-py39-ubuntu20.04-sagemaker		      |
| TensorFlow 2.10.1 |inference	|No			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.1-gpu-py39-cu112-ubuntu20.04-sagemaker	 |
| TensorFlow 2.10.0 |inference	|No			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.0-cpu-py39-ubuntu20.04-sagemaker		      |
| TensorFlow 2.10.0 |inference	|No			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.10.0-gpu-py39-cu112-ubuntu20.04-sagemaker	 |
| TensorFlow 2.9.3 |inference   |No     	|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.3-cpu-py39-ubuntu20.04-sagemaker		      |
| TensorFlow 2.9.3 |inference	|No			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.3-gpu-py39-cu112-ubuntu20.04-sagemaker	 |
| TensorFlow 2.9.2 |training	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9.2-cpu-py39-ubuntu20.04-sagemaker		       |
| TensorFlow 2.9.2 |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.9.2-gpu-py39-cu112-ubuntu20.04-sagemaker	 |
| TensorFlow 2.9.2 |inference   |No     	|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.2-cpu-py39-ubuntu20.04-sagemaker		      |
| TensorFlow 2.9.2 |inference	|No			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.9.2-gpu-py39-cu112-ubuntu20.04-sagemaker	 |
| TensorFlow 2.8.3 |training	|Yes			|CPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.8.3-cpu-py39-ubuntu20.04-sagemaker		       |
| TensorFlow 2.8.3 |training	|Yes			|GPU 		| 3.9 (py39)			| 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.8.3-gpu-py39-cu112-ubuntu20.04-sagemaker	  |

Prior EC2 Framework Graviton Containers
============================

| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	|Example URL																						|
|-------------------|-----------|---------------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|PyTorch 1.12.1   |inference	|No			|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:1.12.1-cpu-py38-ubuntu20.04-ec2		|
|TensorFlow 2.9.1   |inference	|No			|CPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-graviton:2.9.1-cpu-py38-ubuntu20.04-ec2		|

Prior SageMaker Framework Graviton Containers
============================
| Framework         |Job Type	|Horovod Options|CPU/GPU 	|Python Version Options	|Example URL																						|
|-------------------|-----------|---------------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
| PyTorch 1.12.1    | inference | No			 | CPU 	   | 3.8 (py38)			    | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-graviton:1.12.1-cpu-py38-ubuntu20.04-sagemaker   |
| TensorFlow 2.9.1  | inference | No			 | CPU 	   | 3.9 (py39)			    | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-graviton:2.9.1-cpu-py38-ubuntu20.04-sagemaker |

Prior AutoGluon Training Containers
===============================

| Framework       | AutoGluon Version | Job Type | CPU/GPU | Python Version Options | Example URL                                                                                      |
|-----------------|-------------------|----------|---------|------------------------|--------------------------------------------------------------------------------------------------|
| AutoGluon 0.6.2 | 0.6.2             | training | GPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:0.6.2-gpu-py38-cu113-ubuntu20.04 |
| AutoGluon 0.6.2 | 0.6.2             | training | CPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-training:0.6.2-cpu-py38-ubuntu20.04       |

Prior AutoGluon Inference Containers
===============================

| Framework       | AutoGluon Version | Job Type  | CPU/GPU | Python Version Options | Example URL                                                                                       |
|-----------------|-------------------|-----------|---------|------------------------|---------------------------------------------------------------------------------------------------|
| AutoGluon 0.6.2 | 0.6.2             | inference | GPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-inference:0.6.2-gpu-py38-cu113-ubuntu20.04 |
| AutoGluon 0.6.2 | 0.6.2             | inference | CPU     | 3.8 (py38)             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/autogluon-inference:0.6.2-cpu-py38-ubuntu20.04       |

Prior SageMaker Training Compiler Containers
===============================

| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL																						|
|-----------------------------------------------|-----------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|TensorFlow 2.6.3 with HuggingFace transformers 4.17.0 and SageMaker Training Compiler |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-tensorflow-trcomp-training:2.6.3-transformers4.17.0-gpu-py38-cu112-ubuntu20.04 	|
|PyTorch 1.12.0 with SageMaker Training Compiler    |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-trcomp-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker     |

Prior HuggingFace Training Containers
===============================

| Framework                                     |Job Type	|CPU/GPU 	|Python Version Options	|Example URL																						|
|-----------------------------------------------|-----------|-----------|-----------------------|---------------------------------------------------------------------------------------------------|
|PyTorch 1.10.2 with HuggingFace transformers    |training	|GPU 		| 3.8 (py38)			|763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:1.10.2-transformers4.17.0-gpu-py38-cu113-ubuntu20.04      |
