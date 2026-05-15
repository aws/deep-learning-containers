# Image Access

## Public ECR (No Authentication)

All DLC images are available on [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers) and can be pulled without {{ aws }}
credentials:

```bash
docker pull public.ecr.aws/deep-learning-containers/vllm:server-cuda
```

## Private ECR

The same images are also available in private ECR registries (one per region). Authenticate first:

```bash
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-west-2.amazonaws.com
```

Then pull using the private URL format: `<account_id>.dkr.ecr.<region>.amazonaws.com/<repository>:<tag>`

Find account IDs for all regions in the [Region Availability](../reference/region_availability.md) table. Find repository names and tags in the
[Available Images](../reference/available_images.md) table.
