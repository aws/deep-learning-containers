# Please only set it to True if you are preparing an EI related PR
# Do remember to revert it back to False before merging any PR (including EI dedicated PR)
ENABLE_EI_MODE = False
# Frameworks for which you want to disable both builds and tests
DISABLE_FRAMEWORK_TESTS = ["tensorflow", "mxnet"]
# Disable new builds or build without datetime tag
DISABLE_DATETIME_TAG = True
# Note: Need to build the images at least once with DISABLE_DATETIME_TAG = True
# before disabling new builds or tests will fail
DISABLE_NEW_BUILDS = True
