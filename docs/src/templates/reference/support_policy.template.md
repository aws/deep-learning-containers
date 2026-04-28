# Framework Support Policy

[{{ dlc_long }}](../index.md) simplify image configuration for deep learning workloads and are optimized with the latest frameworks, hardware, drivers, libraries, and operating systems. This page details the framework support policy for {{ dlc_short }}s.

## Glossary

- **GA (General Availability)**: The date when a framework version becomes officially supported and available for production use.
- **EOP (End of Patch)**: The date after which a framework version no longer receives security patches or bug fixes.

## Notice

- We cannot guarantee security patching on Ubuntu-based vLLM and SGLang images due to the lack of Ubuntu Pro licensing. Customers may continue using these images at their own discretion and risk. We recommend migrating to our Amazon Linux-based images.
- We are extending support for PyTorch 2.6 Inference images until end of June 2026 as these are the last available PyTorch inference images with torchserve support.

## Supported Frameworks

{{ supported_table }}

## Unsupported Frameworks

{{ unsupported_table }}

## End of Patch Availability

After a framework version reaches its End of Patch (EOP) date, the container images will remain available on {{ ecr }} and, where applicable, the {{ ecr_public }}. However, these images will no longer receive security patches or bug fixes. We recommend upgrading to a supported framework version to continue receiving updates.
