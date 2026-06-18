# Release Schedule

Release window: 9:00 AM – 4:00 PM PDT/PST (16:00 – 23:00 UTC)

## Monday / Wednesday

| Cron            | UTC   | PDT/PST       | Workflow                                  | GPU     |
| --------------- | ----- | ------------- | ----------------------------------------- | ------- |
| `00 16 * * 1,3` | 16:00 | 09:00 / 08:00 | `vllm.autorelease-ec2-amzn2023.yml`       | Heavy   |
| `00 17 * * 1,3` | 17:00 | 10:00 / 09:00 | `pytorch.autorelease-2.11-ec2.yml`        | Light   |
| `00 18 * * 1,3` | 18:00 | 11:00 / 10:00 | `vllm.autorelease-sagemaker-amzn2023.yml` | Heavy   |
| `00 19 * * 1,3` | 19:00 | 12:00 / 11:00 | `pytorch.autorelease-2.11-sagemaker.yml`  | Light   |
| `00 20 * * 1,3` | 20:00 | 13:00 / 12:00 | `vllm.autorelease-ec2-ubuntu.yml`         | Heavy   |
| `00 21 * * 1,3` | 21:00 | 14:00 / 13:00 | `base.autorelease-cu129.yml`              | Minimal |
| `30 21 * * 1,3` | 21:30 | 14:30 / 13:30 | `vllm.autorelease-sagemaker-ubuntu.yml`   | Heavy   |
| `30 22 * * 1,3` | 22:30 | 15:30 / 14:30 | `base.autorelease-cu130.yml`              | Minimal |

## Tuesday / Thursday

| Cron            | UTC   | PDT/PST       | Workflow                                    | GPU    |
| --------------- | ----- | ------------- | ------------------------------------------- | ------ |
| `00 16 * * 2,4` | 16:00 | 09:00 / 08:00 | `sglang.autorelease-ec2-amzn2023.yml`       | Heavy  |
| `00 17 * * 2,4` | 17:00 | 10:00 / 09:00 | `ray.autorelease-ec2.yml`                   | Light  |
| `00 18 * * 2,4` | 18:00 | 11:00 / 10:00 | `sglang.autorelease-sagemaker-amzn2023.yml` | Heavy  |
| `00 19 * * 2,4` | 19:00 | 12:00 / 11:00 | `vllm-omni.autorelease-ec2.yml`             | Medium |
| `00 20 * * 2,4` | 20:00 | 13:00 / 12:00 | `sglang.autorelease-ec2-ubuntu.yml`         | Heavy  |
| `00 21 * * 2,4` | 21:00 | 14:00 / 13:00 | `ray.autorelease-sagemaker.yml`             | Light  |
| `30 21 * * 2,4` | 21:30 | 14:30 / 13:30 | `sglang.autorelease-sagemaker-ubuntu.yml`   | Heavy  |
| `00 22 * * 2,4` | 22:00 | 15:00 / 14:00 | `vllm-omni.autorelease-sagemaker.yml`       | Medium |
