## AWS Neuron Deep Learning Containers

AWS Neuron Deep Learning Containers (DLCs) are a set of Docker images for training and serving models on AWS Trainium and Inferentia instances using AWS Neuron SDK.

## Containers

### pytorch-inference-neuron

| Framework                                                                                                                              | Neuron Package                  | Neuron SDK Version | Supported EC2 Instance Types | Python Version Options | ECR Public URL                                                                           | Other Packages   |
|----------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|--------------------|------------------------------|------------------------|------------------------------------------------------------------------------------------|------------------|
| [PyTorch 1.13.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.18.0/docker/pytorch/inference/1.13.1/Dockerfile.neuron) | aws-neuronx-tools, torch-neuron | Neuron 2.18.0      | inf1                         | 3.10 (py310)           | public.ecr.aws/neuron/pytorch-inference-neuron:1.13.1-neuron-py310-sdk2.18.0-ubuntu20.04 | torchserve 0.9.0 |

### pytorch-inference-neuronx

| Framework                                                                                                                               | Neuron Package                                                              | Neuron SDK Version | Supported EC2 Instance Types | Python Version Options | ECR Public URL                                                                             | Other Packages   |
|-----------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|--------------------|------------------------------|------------------------|--------------------------------------------------------------------------------------------|------------------|
| [PyTorch 2.1.2](https://github.com/aws-neuron/deep-learning-containers/blob/2.18.0/docker/pytorch/inference/2.1.2/Dockerfile.neuronx)   | aws-neuronx-tools, neuronx_distributed, torch-neuronx, transformers-neuronx | Neuron 2.18.0      | trn1,inf2                    | 3.10 (py310)           | public.ecr.aws/neuron/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.18.0-ubuntu20.04  | torchserve 0.9.0 |
| [PyTorch 1.13.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.18.0/docker/pytorch/inference/1.13.1/Dockerfile.neuronx) | aws-neuronx-tools, neuronx_distributed, torch-neuronx, transformers-neuronx | Neuron 2.18.0      | trn1,inf2                    | 3.10 (py310)           | public.ecr.aws/neuron/pytorch-inference-neuronx:1.13.1-neuronx-py310-sdk2.18.0-ubuntu20.04 | torchserve 0.9.0 |

### pytorch-training-neuronx

| Framework                                                                                                                              | Neuron Package                                        | Neuron SDK Version | Supported EC2 Instance Types | Python Version Options | ECR Public URL                                                                            |
|----------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|--------------------|------------------------------|------------------------|-------------------------------------------------------------------------------------------|
| [PyTorch 2.1.2](https://github.com/aws-neuron/deep-learning-containers/blob/2.18.0/docker/pytorch/training/2.1.2/Dockerfile.neuronx)   | aws-neuronx-tools, neuronx_distributed, torch-neuronx | Neuron 2.18.0      | trn1,inf2                    | 3.10 (py310)           | public.ecr.aws/neuron/pytorch-training-neuronx:2.1.2-neuronx-py310-sdk2.18.0-ubuntu20.04  |
| [PyTorch 1.13.1](https://github.com/aws-neuron/deep-learning-containers/blob/2.18.0/docker/pytorch/training/1.13.1/Dockerfile.neuronx) | aws-neuronx-tools, neuronx_distributed, torch-neuronx | Neuron 2.18.0      | trn1,inf2                    | 3.10 (py310)           | public.ecr.aws/neuron/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.18.0-ubuntu20.04 |

## Security

See [SECURITY](SECURITY.md) for more information.

## License

This project is licensed under the Apache-2.0 License.
