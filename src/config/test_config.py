# Please only set it to True if you are preparing a Benchmark related PR
# Do remember to revert it back to False before merging any PR (including Benchmark dedicated PR)
ENABLE_BENCHMARK_DEV_MODE = True
# Disable the test codebuild jobs to be run
DISABLE_SANITY_TESTS = True
DISABLE_SAGEMAKER_TESTS = False
DISABLE_ECS_TESTS = True
DISABLE_EKS_TESTS = True
DISABLE_EC2_TESTS = True
USE_SCHEDULER = False
