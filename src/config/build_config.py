# Frameworks for which you want to disable both builds and tests (General/Non-EI purpose)
DISABLE_FRAMEWORK_TESTS = []
# Frameworks for which you want to disable both builds and tests (EI purpose)
DISABLE_EI_FRAMEWORK_TESTS = ["tensorflow", "mxnet", "pytorch"]
# Disable new builds or build without datetime tag
DISABLE_DATETIME_TAG = False
# Note: Need to build the images at least once with DISABLE_DATETIME_TAG = True
# before disabling new builds or tests will fail
DISABLE_NEW_BUILDS = False
