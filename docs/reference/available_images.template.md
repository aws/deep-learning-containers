# Available Deep Learning Containers Images

Replace `<repository-name>` and `<image-tag>` based on your desired container.

## Getting Started

Once you've selected your desired Deep Learning Containers image, continue with one of the following:

- [Amazon EC2 Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ec2.html)
- [Amazon ECS Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-ecs.html)
- [Amazon EKS Tutorials](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-eks.html)
- [Hugging Face on AWS](https://huggingface.co/docs/sagemaker/en/index)
- [Security in AWS Deep Learning Containers](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/security.html)
- [Release Notes](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/dlc-release-notes.html)

## Region Availability

| Region                    | Code           | General | Neuron | Example URL                                                                          |
| ------------------------- | -------------- | ------- | ------ | ------------------------------------------------------------------------------------ |
| US East (Ohio)            | us-east-2      | ✅      | ✅     | `763104351884.dkr.ecr.us-east-2.amazonaws.com/<repository-name>:<image-tag>`         |
| US East (N. Virginia)     | us-east-1      | ✅      | ✅     | `763104351884.dkr.ecr.us-east-1.amazonaws.com/<repository-name>:<image-tag>`         |
| US West (N. California)   | us-west-1      | ✅      | ❌     | `763104351884.dkr.ecr.us-west-1.amazonaws.com/<repository-name>:<image-tag>`         |
| US West (Oregon)          | us-west-2      | ✅      | ✅     | `763104351884.dkr.ecr.us-west-2.amazonaws.com/<repository-name>:<image-tag>`         |
| Africa (Cape Town)        | af-south-1     | ✅      | ❌     | `626614931356.dkr.ecr.af-south-1.amazonaws.com/<repository-name>:<image-tag>`        |
| Asia Pacific (Hong Kong)  | ap-east-1      | ✅      | ❌     | `871362719292.dkr.ecr.ap-east-1.amazonaws.com/<repository-name>:<image-tag>`         |
| Asia Pacific (Hyderabad)  | ap-south-2     | ✅      | ❌     | `772153158452.dkr.ecr.ap-south-2.amazonaws.com/<repository-name>:<image-tag>`        |
| Asia Pacific (Jakarta)    | ap-southeast-3 | ✅      | ❌     | `907027046896.dkr.ecr.ap-southeast-3.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Malaysia)   | ap-southeast-5 | ✅      | ❌     | `550225433462.dkr.ecr.ap-southeast-5.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Melbourne)  | ap-southeast-4 | ✅      | ❌     | `457447274322.dkr.ecr.ap-southeast-4.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Mumbai)     | ap-south-1     | ✅      | ✅     | `763104351884.dkr.ecr.ap-south-1.amazonaws.com/<repository-name>:<image-tag>`        |
| Asia Pacific (Osaka)      | ap-northeast-3 | ✅      | ❌     | `364406365360.dkr.ecr.ap-northeast-3.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Seoul)      | ap-northeast-2 | ✅      | ❌     | `763104351884.dkr.ecr.ap-northeast-2.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Singapore)  | ap-southeast-1 | ✅      | ✅     | `763104351884.dkr.ecr.ap-southeast-1.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Sydney)     | ap-southeast-2 | ✅      | ✅     | `763104351884.dkr.ecr.ap-southeast-2.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Taipei)     | ap-east-2      | ✅      | ✅     | `763104351884.dkr.ecr.ap-east-2.amazonaws.com/<repository-name>:<image-tag>`         |
| Asia Pacific (Thailand)   | ap-southeast-7 | ✅      | ❌     | `590183813437.dkr.ecr.ap-southeast-7.amazonaws.com/<repository-name>:<image-tag>`    |
| Asia Pacific (Tokyo)      | ap-northeast-1 | ✅      | ✅     | `763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/<repository-name>:<image-tag>`    |
| Canada (Central)          | ca-central-1   | ✅      | ❌     | `763104351884.dkr.ecr.ca-central-1.amazonaws.com/<repository-name>:<image-tag>`      |
| Canada (Calgary)          | ca-west-1      | ✅      | ❌     | `204538143572.dkr.ecr.ca-west-1.amazonaws.com/<repository-name>:<image-tag>`         |
| EU (Frankfurt)            | eu-central-1   | ✅      | ✅     | `763104351884.dkr.ecr.eu-central-1.amazonaws.com/<repository-name>:<image-tag>`      |
| EU (Ireland)              | eu-west-1      | ✅      | ✅     | `763104351884.dkr.ecr.eu-west-1.amazonaws.com/<repository-name>:<image-tag>`         |
| EU (London)               | eu-west-2      | ✅      | ❌     | `763104351884.dkr.ecr.eu-west-2.amazonaws.com/<repository-name>:<image-tag>`         |
| EU (Milan)                | eu-south-1     | ✅      | ❌     | `692866216735.dkr.ecr.eu-south-1.amazonaws.com/<repository-name>:<image-tag>`        |
| EU (Paris)                | eu-west-3      | ✅      | ✅     | `763104351884.dkr.ecr.eu-west-3.amazonaws.com/<repository-name>:<image-tag>`         |
| EU (Spain)                | eu-south-2     | ✅      | ❌     | `503227376785.dkr.ecr.eu-south-2.amazonaws.com/<repository-name>:<image-tag>`        |
| EU (Stockholm)            | eu-north-1     | ✅      | ❌     | `763104351884.dkr.ecr.eu-north-1.amazonaws.com/<repository-name>:<image-tag>`        |
| EU (Zurich)               | eu-central-2   | ✅      | ❌     | `380420809688.dkr.ecr.eu-central-2.amazonaws.com/<repository-name>:<image-tag>`      |
| Israel (Tel Aviv)         | il-central-1   | ✅      | ❌     | `780543022126.dkr.ecr.il-central-1.amazonaws.com/<repository-name>:<image-tag>`      |
| Mexico (Central)          | mx-central-1   | ✅      | ❌     | `637423239942.dkr.ecr.mx-central-1.amazonaws.com/<repository-name>:<image-tag>`      |
| Middle East (Bahrain)     | me-south-1     | ✅      | ❌     | `217643126080.dkr.ecr.me-south-1.amazonaws.com/<repository-name>:<image-tag>`        |
| Middle East (UAE)         | me-central-1   | ✅      | ❌     | `914824155844.dkr.ecr.me-central-1.amazonaws.com/<repository-name>:<image-tag>`      |
| South America (Sao Paulo) | sa-east-1      | ✅      | ✅     | `763104351884.dkr.ecr.sa-east-1.amazonaws.com/<repository-name>:<image-tag>`         |
| China (Beijing)           | cn-north-1     | ✅      | ❌     | `727897471807.dkr.ecr.cn-north-1.amazonaws.com.cn/<repository-name>:<image-tag>`     |
| China (Ningxia)           | cn-northwest-1 | ✅      | ❌     | `727897471807.dkr.ecr.cn-northwest-1.amazonaws.com.cn/<repository-name>:<image-tag>` |

## Authentication

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

Then pull images:

```bash
docker pull <name of container image>
```

## Image Tag Guide

Select your framework, job type (`training`, `inference`, or `general`), and Python version (`py38`, `py39`, `py310`, `py311`, or `py312`).

Pin versions by adding the version tag:

```
763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.4.1-cpu-py37-ubuntu18.04-v1.0
```

---
