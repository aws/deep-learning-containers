# Frameworks for which you want to disable both builds and tests
DISABLE_FRAMEWORK_TESTS = ["pytorch","mxnet"]
# Disable new builds or build without datetime tag
DISABLE_DATETIME_TAG = True
# Note: Need to build the images at least once with DISABLE_DATETIME_TAG = True
# before disabling new builds or tests will fail
DISABLE_NEW_BUILDS = False
