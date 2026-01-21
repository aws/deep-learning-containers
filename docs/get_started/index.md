# Getting Started

Get up and running with {{ dlc_long }} quickly.

## Prerequisites

- An {{ aws }} account with appropriate permissions
- Docker installed on your local machine
- {{ aws }} CLI configured with your credentials

## Pulling Images

Learn how to authenticate and pull {{ dlc_long }} images.

### Authentication

```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com
```

Then pull images:

```bash
docker pull <account_id>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>
```

### Image URL Format

To form your container image URL, use the following format:

```
<account_id>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>
```

Where:

- `<account_id>`: Find the account ID for your region in the [Region Availability](../reference/available_images.md#region-availability) table
- `<region>`: Your {{ aws }} region (e.g., `us-east-1`, `us-west-2`, `eu-west-1`)
- `<repository>`: The framework repository name (e.g., `pytorch-training`, `tensorflow-inference`)
- `<tag>`: The image tag from the [Available Images](../reference/available_images.md) tables

### Example

```bash
# Authenticate
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com

# Pull PyTorch training image
docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/{{ images.latest_pytorch_training_ec2 }}
```

## Next Steps

After completing the getting started guides, explore our [tutorials](../tutorials/index.md) for more advanced use cases.
