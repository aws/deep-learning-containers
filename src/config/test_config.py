# Please only set it to True if you are preparing a Benchmark related PR
# Do remember to revert it back to False before merging any PR (including Benchmark dedicated PR)
ENABLE_BENCHMARK_DEV_MODE = False

# Disable the test codebuild jobs to be run

# It is recommended to set DISABLE_EFA_TESTS to True to disable EFA tests if there is no change to EFA installer version or Frameworks.
DISABLE_EFA_TESTS = False

DISABLE_SANITY_TESTS = False
DISABLE_SAGEMAKER_TESTS = True
DISABLE_ECS_TESTS = True
DISABLE_EKS_TESTS = True
DISABLE_EC2_TESTS = True
USE_SCHEDULER = False
