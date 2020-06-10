# Available Deep Learning Containers Images

The following table lists the Docker image URLs that will be used by Amazon ECS in task definitions. Replace the `<repository-name>` and `<image-tag>` values based on your desired container.

Once you've selected your desired Deep Learning Containers image, continue with the one of the following:

-   To run training and inference on Deep Learning Containers for Amazon EC2 using MXNet, PyTorch, TensorFlow, and TensorFlow 2, see [Amazon EC2 Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2.html)

-   To run training and inference on Deep Learning Containers for Amazon ECS using MXNet, PyTorch, and TensorFlow, see [Amazon ECS tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs.html)

-   Deep Learning Containers for Amazon EKS offer CPU, GPU, and distributed GPU-based training, as well as CPU and GPU-based inference. To run training and inference on Deep Learning Containers for Amazon EKS using MXNet, PyTorch, and TensorFlow, see [Amazon EKS Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-eks.html)

-   For information on security in Deep Learning Containers, see [Security in AWS Deep Learning Containers](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/security.html)

-   For a list of the latest Deep Learning Containers release notes, see [Release Notes for Deep Learning Containers](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-release-notes.html)


Deep Learning Containers Docker Images are available in the following regions:
| Region 			                      |  			 Code 			           |  			 General Container 			 |  			 Elastic Inference Container 			 |  			 Example URL 			                                                                       |
|------------------------------|-------------------|----------------------|--------------------------------|--------------------------------------------------------------------------------------|
|  			 US East (N. Virginia) 			     |  			 us-east-1 			      |  			 Available 			         |  			 Available 			                   |  			 763104351884.dkr.ecr.us-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			        |
|  			 US East (Ohio) 			            |  			 us-east-2 			      |  			 Available 			         |  			 Available 			                   |  			 763104351884.dkr.ecr.us-east-2.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			        |
|  			 US West (N. California) 			   |  			 us-west-1 			      |  			 Available 			         |  			 None 			                        |  			 763104351884.dkr.ecr.us-west-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			        |
|  			 US West (Oregon) 			          |  			 us-west-2 			      |  			 Available 			         |  			 Available 			                   |  			 763104351884.dkr.ecr.us-west-2.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			        |
|  			 Asia Pacific (Tokyo) 			      |  			 ap-northeast-1 			 |  			 Available 			         |  			 Available 			                   |  			 763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			   |
|  			 Asia Pacific (Seoul) 			      |  			 ap-northeast-2 			 |  			 Available 			         |  			 Available 			                   |  			 763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			   |
|  			 Asia Pacific (Hong Kong) 			  |  			 ap-east-1 			      |  			 Available 			         |  			 None 			                        |  			 871362719292.dkr.ecr.ap-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			        |
|  			 Asia Pacific (Mumbai) 			     |  			 ap-south-1 			     |  			 Available 			         |  			 None 			                        |  			 763104351884.dkr.ecr.ap-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			       |
|  			 Asia Pacific (Singapore) 			  |  			 ap-southeast-1 			 |  			 Available 			         |  			 None 			                        |  			 763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			   |
|  			 Asia Pacific (Sydney) 			     |  			 ap-southeast-2 			 |  			 Available 			         |  			 None 			                        |  			 763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			   |
|  			 Middle East (Bahrain) 			     |  			 me-south-1 			     |  			 Available 			         |  			 None 			                        |  			 217643126080.dkr.ecr.me-south-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			       |
|  			 South America (Sao Paulo) 			 |  			 sa-east-1 			      |  			 Available 			         |  			 None 			                        |  			 763104351884.dkr.ecr.sa-east-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			        |
|  			 Canada (Central) 			          |  			 ca-central-1 			   |  			 Available 			         |  			 None 			                        |  			 763104351884.dkr.ecr.ca-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			     |
|  			 EU (Frankfurt) 			            |  			 eu-central-1 			   |  			 Available 			         |  			 None 			                        |  			 763104351884.dkr.ecr.eu-central-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			     |
|  			 EU (Stockholm) 			            |  			 eu-north-1 			     |  			 Available 			         |  			 None 			                        |  			 763104351884.dkr.ecr.eu-north-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			       |
|  			 EU (Ireland) 			              |  			 eu-west-1 			      |  			 Available 			         |  			 Available 			                   |  			 763104351884.dkr.ecr.eu-west-1.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			        |
|  			 EU (London) 			               |  			 eu-west-2 			      |  			 Available 			         |  			 None 			                        |  			 763104351884.dkr.ecr.eu-west-2.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			        |
|  			 EU (Paris) 			                |  			 eu-west-3 			      |  			 Available 			         |  			 None 			                        |  			 763104351884.dkr.ecr.eu-west-3.amazonaws.com/&lt;repository-name>:&lt;image-tag> 			        |
|  			 China (Beijing) 			           |  			 cn-north-1 			     |  			 Available 			         |  			 None 			                        |  			 727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/&lt;repository-name>:&lt;image-tag> 			    |
|  			 China (Ningxia) 			           |  			 cn-northwest-1 			 |  			 Available 			         |  			 None 			                        |  			 727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/&lt;repository-name>:&lt;image-tag> |


Note: **eu-north-1** is available starting in the version 2.0 release.
As a result, MXNet-1.4.0 is not available in this region.

ECR is a regional service and the Image table contains the URLs for
**us-east-1** images. To pull from one of the regions mentioned
previously, insert the region in the repository URL following this
example:



     763104351884.dkr.ecr.<region>.amazonaws.com/tensorflow-training:1.15.2-cpu-py27-ubuntu18.04

**Important**

You must login to access to the DLC image repository before pulling
the image. Ensure your CLI is up to date using the steps in [Installing the current AWS CLI Version](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv1.html#install-tool-bundled)
    Then, specify your region and its corresponding ECR Registry from
    the previous table in the following command:



        aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

You can then pull these Docker images from ECR by running:



    docker pull <name of container image>

 General Framework Containers
============================

To use the following table, select your desired framework, as well as
the kind of job you're starting, and the desired Python version. Your
job type is either ``training`` or ``inference``. Your Python version is
either ``py27`` or ``py36``. Plug this information into the replaceable
portions of the URL as shown in the example URL.

| Framework 			        |  			 Job Type 			            |  			 Horovod Options 			              |  			 CPU/GPU 			 |  			 Python Version Options 			 |  			 Example URL 			                                                                                     |
|-------------------|------------------------|---------------------------------|------------|---------------------------|----------------------------------------------------------------------------------------------------|
|  			 TensorFlow 2.2 			 |  			 training 			            | Yes                             |  			 CPU 			     |  			 3.7 (py37) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.2.0-cpu-py37-ubuntu18.04        |
|  			 TensorFlow 2.1 			 |  			 inference 			           | No                              |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			 | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.1.0-cpu-py27-ubuntu18.04       |
|  			 TensorFlow 2.2 			 |  			 training 			            | Yes                             |  			 GPU 			     |  			 3.7 (py37) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.2.0-gpu-py37-cu101-ubuntu18.04  |
|  			 TensorFlow 2.1 			 |  			 inference 			           |  			 No 			                           |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			 | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.1.0-gpu-py27-cu101-ubuntu18.04 |
|  			 MXNet 1.6.0 			    |  			 training, inference 			 |  			 Training: Yes, Inference: No 			 |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			 | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.6.0-cpu-py27-ubuntu16.04             |
|  			 MXNet 1.6.0 			    |  			 training, inference 			 |  			 Training: Yes, Inference: No 			 |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			 | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.6.0-gpu-py27-cu101-ubuntu16.04       |
| PyTorch 1.5.0     |  			 training 			            | No                              |  			 CPU 			     |  			 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.5.0-cpu-py36-ubuntu16.04           |
| PyTorch 1.5.0     |  			 inference 			           | No                              |  			 CPU 			     |  			 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.5.0-cpu-py36-ubuntu16.04          |
| PyTorch 1.5.0     |  			 training 			            |  			 Yes 			                          |  			 GPU 			     |  			 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.5.0-gpu-py36-cu101-ubuntu16.04     |
| PyTorch 1.5.0     |  			 inference 			           |  			 No 			                           |  			 GPU 			     |  			 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.5.0-gpu-py36-cu101-ubuntu16.04    |


Elastic Inference Containers
============================

| Framework 			                                  |  			 Job Type 			  |  			 Horovod Options 			 |  			 CPU/GPU 			 |  			 Python Version Options 			 |  			 Example URL 			                                                                                    |
|---------------------------------------------|--------------|--------------------|------------|---------------------------|---------------------------------------------------------------------------------------------------|
|  			 TensorFlow 1.15.0 with Elastic Inference 			 |  			 inference 			 |  			 No 			              |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			 | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-eia:1.15.0-cpu-py27-ubuntu18.04 |
|  			 TensorFlow 2.0.0 with Elastic Inference 			  |  			 inference 			 |  			 No 			              |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			 | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-eia:2.0.0-cpu-py27-ubuntu18.04  |
|  			 MXNet 1.5.1 with Elastic Inference 			       |  			 inference 			 |  			 No 			              |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			 | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference-eia:1.5.1-cpu-py27-ubuntu16.04       |
|  			 PyTorch 1.3.1 with Elastic Inference 			     |  			 inference 			 |  			 No 			              |  			 CPU 			     |  			 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-eia:1.3.1-cpu-py36-ubuntu16.04     |


Prior Versions
==============

| Framework 			                                  |  			 Job Type 			            |  			 Horovod Options 			              |  			 CPU/GPU 			 |  			 Python Version Options 			             |  			 Example URL 			                                                                                     |
|---------------------------------------------|------------------------|---------------------------------|------------|---------------------------------------|----------------------------------------------------------------------------------------------------|
|  			 TensorFlow 2.1 			                           |  			 training 			            | No                              |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.1.0-cpu-py27-ubuntu18.04        |
|  			 TensorFlow 2.1 			                           |  			 training 			            |  			 No 			                           |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.1.0-gpu-py27-cu101-ubuntu18.04  |
|  			 TensorFlow 1.15.2 			                        |  			 training 			            | Yes                             |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36), 3.7 (py37) 			 | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.2-cpu-py27-ubuntu18.04       |
|  			 TensorFlow 1.15.2 			                        |  			 inference 			           | No                              |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.2-cpu-py27-ubuntu18.04       |
|  			 TensorFlow 1.15.2 			                        |  			 training 			            |  			 Yes 			                          |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36), 3.7 (py37) 			 | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.2-gpu-py27-cu100-ubuntu18.04 |
|  			 TensorFlow 1.15.2 			                        |  			 inference 			           |  			 No 			                           |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.2-gpu-py27-cu100-ubuntu18.04 |
|  			 MXNet 1.4.1 with Elastic Inference 			       |  			 inference 			           |  			 No 			                           |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference-eia:1.4.1-cpu-py27-ubuntu16.04        |
| PyTorch 1.4.0                               |  			 training 			            | No                              |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.4.0-cpu-py27-ubuntu16.04           |
| PyTorch 1.4.0                               |  			 inference 			           | No                              |  			 CPU 			     |  			 3.6 (py36) 			                         | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.4.0-cpu-py36-ubuntu16.04          |
| PyTorch 1.4.0                               |  			 training 			            |  			 Yes 			                          |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.4.0-gpu-py27-cu101-ubuntu16.04     |
| PyTorch 1.4.0                               |  			 inference 			           |  			 No 			                           |  			 GPU 			     |  			 3.6 (py36) 			                         | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.4.0-gpu-py36-cu101-ubuntu16.04    |
|  			 TensorFlow 1.14.0 with Elastic Inference 			 |  			 inference 			           |  			 No 			                           |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference-eia:1.14.0-cpu-py27-ubuntu16.04  |
|  			 TensorFlow 2.0.1 			                         |  			 training, inference 			 | Training:Yes, Inference: No     |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.0.1-cpu-py27-ubuntu18.04        |
|  			 TensorFlow 2.0.1 			                         | training, inference    |  			 Training: Yes, Inference: No 			 |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.0.1-gpu-py27-cu100-ubuntu18.04  |
|  			 TensorFlow 2.0.0 			                         |  			 training, inference 			 | Training:Yes, Inference: No     |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.0.0-cpu-py27-ubuntu18.04        |
|  			 TensorFlow 2.0.0 			                         |  			 training, inference 			 |  			 Training: Yes, Inference: No 			 |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.0.0-gpu-py27-cu100-ubuntu18.04  |
|  			 TensorFlow 1.15.0 			                        | training, inference    | Training:Yes, Inference: No     |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.0-cpu-py27-ubuntu18.04       |
|  			 TensorFlow 1.15.0 			                        |  			 training, inference 			 |  			 Training: Yes, Inference: No 			 |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.0-gpu-py27-cu100-ubuntu18.04 |
|  			 TensorFlow 1.14.0 			                        |  			 training, inference 			 | Training:Yes, Inference: No     |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.14.0-cpu-py27-ubuntu16.04       |
|  			 TensorFlow 1.14.0 			                        |  			 training, inference 			 |  			 Training: Yes, Inference: No 			 |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.14.0-gpu-py27-cu100-ubuntu16.04 |
|  			 MXNet 1.4.1 			                              |  			 training, inference 			 |  			 No 			                           |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.4.1-cpu-py27-ubuntu16.04             |
|  			 MXNet 1.4.1 			                              |  			 training, inference 			 |  			 No 			                           |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.4.1-gpu-py27-cu100-ubuntu16.04       |
| PyTorch 1.3.1                               |  			 training, inference 			 |  			 No 			                           |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.3.1-cpu-py27-ubuntu16.04           |
| PyTorch 1.3.1                               |  			 training, inference 			 |  			 Training:Yes, Inference: No 			  |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.3.1-gpu-py27-cu101-ubuntu16.04     |
| PyTorch 1.2.0                               |  			 training, inference 			 | No                              |  			 CPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.2.0-cpu-py27-ubuntu16.04           |
| PyTorch 1.2.0                               |  			 training, inference 			 |  			 Training:Yes, Inference: No 			  |  			 GPU 			     |  			 2.7 (py27), 3.6 (py36) 			             | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.2.0-gpu-py27-cu100-ubuntu16.04     |
