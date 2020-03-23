# List of safety check issue IDs to ignore. To get issue ID, run safety check on the container,
# and copy paste the ID given by the safety check for a package.
# Note:- 1. This ONLY needs to be done if a package version exists that resolves this safety issue, but that version
#           cannot be used because of incompatibilities.
#        2. Ensure that IGNORE_SAFETY_IDS is always as small/empty as possible upon every release.
IGNORE_SAFETY_IDS = {
    "mxnet-inference": {
        "py2": [],
        "py3": []
    },
    "mxnet-inference-eia": {
        # numpy<=1.16.0 -- This has to only be here while we publish MXNet 1.4.1 EI DLC v1.0
        "py2": ['36810'],
        "py3": ['36810']
    },
    "mxnet-training": {
        "py2": [],
        "py3": []
    },
    "pytorch-inference": {
        "py2": [],
        "py3": []
    },
    "pytorch-inference-eia": {
        "py2": [],
        "py3": []
    },
    "pytorch-training": {
        # astropy<3.0.1
        "py2": ['35810'],
        "py3": []
    },
    "tensorflow-inference": {
        "py2": [],
        "py3": []
    },
    "tensorflow-inference-eia": {
        "py2": [],
        "py3": []
    },
    "tensorflow-training": {
        "py2": [],
        "py3": []
    }
}


def get_safety_ignore_list(repo_name, python_version):
    """
    Get a list of known safety check issue IDs to ignore, if specified in IGNORE_LISTS.
    :param repo_name:
    :param python_version:
    :return: <list> list of safety check IDs to ignore
    """
    return IGNORE_SAFETY_IDS.get(repo_name, {}).get(python_version)
