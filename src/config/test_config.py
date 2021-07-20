from . import parse_dlc_developer_configs

ENABLE_BENCHMARK_DEV_MODE = parse_dlc_developer_configs("dev", "benchmark_mode")

# Disable the test codebuild jobs to be run

# It is recommended to set DISABLE_EFA_TESTS to True
# to disable EFA tests if there are no changes to EFA installer version or Frameworks.
DISABLE_EFA_TESTS = not parse_dlc_developer_configs("test", "efa_tests")
DISABLE_SANITY_TESTS = not parse_dlc_developer_configs("test", "sanity_tests")
DISABLE_SAGEMAKER_TESTS = not parse_dlc_developer_configs("test", "sagemaker_tests")
DISABLE_ECS_TESTS = not parse_dlc_developer_configs("test", "ecs_tests")
DISABLE_EKS_TESTS = not parse_dlc_developer_configs("test", "eks_tests")
DISABLE_EC2_TESTS = not parse_dlc_developer_configs("test", "ec2_tests")
USE_SCHEDULER = parse_dlc_developer_configs("test", "use_scheduler")
