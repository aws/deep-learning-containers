# Framework Support Policy

For more details see the [Framework Support Policy](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/support-policy.html).

## Supported Framework Versions

| Framework  | Version | CUDA        | GitHub GA  | End of Patch |
| ---------- | ------- | ----------- | ---------- | ------------ |
| Base       | -       | 13.0        | 2025-10-22 | 2026-10-22   |
| Base       | -       | 12.9        | 2025-08-18 | 2027-08-18   |
| Base       | -       | 12.8        | 2025-06-05 | 2026-06-05   |
| PyTorch    | 2.9     | 13.0        | 2025-10-15 | 2026-10-15   |
| PyTorch    | 2.8     | 12.9        | 2025-08-06 | 2026-08-06   |
| PyTorch    | 2.7     | 12.8        | 2025-04-23 | 2026-04-23   |
| PyTorch    | 2.6     | 12.6/12.4\* | 2025-01-29 | 2026-01-29   |
| TensorFlow | 2.19    | 12.5/12.2\* | 2025-03-11 | 2026-03-11   |

\*Training/Inference DLCs use different CUDA versions

## Unsupported Framework Versions

Versions listed here will appear for 2 years past their support date.

| Framework  | Version | CUDA        | GitHub GA  | End of Patch |
| ---------- | ------- | ----------- | ---------- | ------------ |
| PyTorch    | 2.5     | 12.4        | 2024-10-29 | 2025-10-29   |
| PyTorch    | 2.4     | 12.4        | 2024-07-24 | 2025-07-24   |
| PyTorch    | 2.3     | 12.1        | 2024-04-24 | 2025-04-24   |
| PyTorch    | 2.2     | 12.1/11.8\* | 2024-01-30 | 2025-01-30   |
| PyTorch    | 2.1     | 12.1/11.8\* | 2023-10-04 | 2024-10-04   |
| PyTorch    | 2.0     | 12.1/11.8\* | 2023-03-15 | 2024-03-15   |
| PyTorch    | 1.13    | 11.7        | 2022-10-28 | 2024-10-28   |
| TensorFlow | 2.18    | 12.5/12.2\* | 2024-10-24 | 2025-10-24   |
| TensorFlow | 2.16    | 12.3/12.2\* | 2024-03-07 | 2025-03-07   |
| TensorFlow | 2.14    | 11.8        | 2023-09-26 | 2024-09-26   |

## Updates

- **2024-08-14**: Updated version format from major.minor.patch to major.minor
- **2023-10-30**: Extended support for PT 1.13 by one year (last 1.x version)
- **2023-10-01**: End of support for MXNet and Elastic Inference

## Unsupported Frameworks

From October 1, 2023, MXNet and Elastic Inference frameworks are not supported due to lack of upstream support. Please use one of the supported frameworks listed above.
