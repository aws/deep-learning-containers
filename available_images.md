# Available Deep Learning Containers Images

> **📢 Announcement:** We have transitioned to automated documentation generation. The available images documentation is now hosted at:
>
> **https://aws.github.io/deep-learning-containers/reference/available_images/**

## Updating Documentation

To add or update images, submit a PR to the `main` branch with a YAML config file:

1. Create a YAML file in [docs/src/data/](https://github.com/aws/deep-learning-containers/tree/main/docs/src/data)`<your-repository>/`
   - File naming is for organization only, but please use a consistent convention for maintenance: `<version>-<accelerator>-<platform>.yml`
   - Examples: [2.9-gpu-ec2.yml](https://github.com/aws/deep-learning-containers/blob/main/docs/src/data/pytorch-training/2.9-gpu-ec2.yml), [2.6-cpu-sagemaker.yml](https://github.com/aws/deep-learning-containers/blob/main/docs/src/data/pytorch-training/2.6-cpu-sagemaker.yml)

2. Include fields that will display in your image tables:
   ```yaml
   framework: PyTorch             # Display name
   version: "2.9"                 # Framework version (quoted)
   accelerator: gpu               # gpu, cpu, or neuronx
   python: py312                  # Python version
   platform: ec2                  # ec2 or sagemaker
   tags:
     - "2.9.0-gpu-py312-cu130-ubuntu22.04-ec2"  # Docker image tag(s)
   ```
   For specific table fields, check out [docs/src/tables/](https://github.com/aws/deep-learning-containers/tree/main/docs/src/tables)

3. Submit a PR to the `main` branch of `aws/deep-learning-containers`

**References:**
- [Full template](https://github.com/aws/deep-learning-containers/blob/main/docs/src/data/template/image-template.yml)
- [Development guide](https://github.com/aws/deep-learning-containers/blob/main/docs/DEVELOPMENT.md)
