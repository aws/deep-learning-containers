# Monitoring and Usage Tracking in AWS Deep Learning Containers

Your AWS Deep Learning Containers do not come with monitoring utilities. For information on monitoring, see [GPU Monitoring and Optimization](https://docs.aws.amazon.com/dlami/latest/devguide/tutorial-gpu.html), [Monitoring Amazon EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/monitoring_ec2.html), [Monitoring Amazon ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-logging-monitoring.html), [Monitoring Amazon EKS](https://docs.aws.amazon.com/eks/latest/userguide/logging-monitoring.html), and [Monitoring Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-incident-response.html).

## Usage Tracking

AWS uses customer feedback and usage information to improve the quality of the services and software we offer to customers. We have added usage data collection to the supported AWS Deep Learning Containers in order to better understand customer usage and guide future improvements. Usage tracking for Deep Learning Containers is activated by default. Customers can change their settings at any point of time to activate or deactivate usage tracking.

Usage tracking for AWS Deep Learning Containers collects the *instance ID*, *frameworks*, *framework versions*, *container types*, and *Python versions* used for the containers. AWS also logs the event time in which it receives this metadata.

No information on the commands used within the containers is collected or retained. No other information about the containers is collected or retained.

To opt out of usage tracking, set the `OPT_OUT_TRACKING` environment variable to true.

```bash
OPT_OUT_TRACKING=true
```

## Failure Rate Tracking

When using a first-party Amazon SageMaker AI AWS Deep Learning Containers [container](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#sagemaker-framework-containers-sm-support-only), the SageMaker AI team will collect failure rate metadata to improve the quality of AWS Deep Learning Containers. Failure rate tracking for AWS Deep Learning Containers is active by default. Customers can change their settings to activate or deactivate failure rate tracking when creating an Amazon SageMaker AI endpoint.

Failure rate tracking for AWS Deep Learning Containers collects the *Instance ID*, *ModelServer name*, *ModelServer version*, *ErrorType*, and *ErrorCode*. AWS also logs the event time in which it receives this metadata.

No information on the commands used within the containers is collected or retained. No other information about the containers is collected or retained.

To opt out of failure rate tracking, set the `OPT_OUT_TRACKING` environment variable to `true`.

```bash
OPT_OUT_TRACKING=true
```

## Usage Tracking in the following Framework Versions

While we recommend updating to supported Deep Learning Containers, to opt-out of usage tracking for Deep Learning Containers that use these frameworks, set the `OPT_OUT_TRACKING` environment variable to true **and** use a custom entry point to disable the call for the following services:

- [Amazon EC2 Custom Entrypoints](https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-ec2-tutorials-custom-entry.html)
- [Amazon ECS Custom Entrypoints](https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-ecs-tutorials-custom-entry.html)
- [Amazon EKS Custom Entrypoints](https://docs.aws.amazon.com/dlami/latest/devguide/deep-learning-containers-eks-tutorials-custom-entry.html)
