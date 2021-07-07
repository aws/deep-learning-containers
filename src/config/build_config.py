from . import parse_dlc_developer_configs

# Please only set it to True if you are preparing an EI related PR
# Do remember to revert it back to False before merging any PR (including EI dedicated PR)
ENABLE_EI_MODE = parse_dlc_developer_configs("dev", "ei_mode")
# Please only set it to True if you are preparing an NEURON related PR
# Do remember to revert it back to False before merging any PR (including NEURON dedicated PR)
ENABLE_NEURON_MODE = parse_dlc_developer_configs("dev", "neuron_mode")
# Frameworks for which you want to disable both builds and tests
DISABLE_FRAMEWORK_TESTS = parse_dlc_developer_configs("build", "skip_frameworks")
# Disable new builds or build without datetime tag
DISABLE_DATETIME_TAG = not parse_dlc_developer_configs("build", "datetime_tag")
# Note: Need to build the images at least once with DISABLE_DATETIME_TAG = True
# before disabling new builds or tests will fail
DISABLE_NEW_BUILDS = not parse_dlc_developer_configs("build", "do_build")
