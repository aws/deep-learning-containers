# Please only enable it if you are preparing an EIA related PR
# Do remember to revert it back before merging any PR (including EIA dedicated PR)
ENABLE_EI_MODE = True
# Frameworks for which you want to disable both builds and tests
DISABLE_FRAMEWORK_TESTS = []
# Disable new builds or build without datetime tag
DISABLE_DATETIME_TAG = False
# Note: Need to build the images at least once with DISABLE_DATETIME_TAG = True
# before disabling new builds or tests will fail
DISABLE_NEW_BUILDS = False
