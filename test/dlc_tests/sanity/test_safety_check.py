##################################################################################################
#
# NOTE: IMPORTANT
# ./test/dlc_tests/sanity/test_safety_check.py is obsolete and no longer used.
#
##################################################################################################
import json
import logging
import os
import sys
import re
import subprocess
import botocore
import boto3
import time

from packaging.specifiers import SpecifierSet
from packaging.version import Version

import pytest
import requests

from invoke import run

from urllib3.util.retry import Retry
from invoke.context import Context
from botocore.exceptions import ClientError

from src.buildspec import Buildspec
import src.utils as src_utils
from test.test_utils import (
    LOGGER,
    CONTAINER_TESTS_PREFIX,
    ec2,
    get_container_name,
    get_framework_and_version_from_tag,
    get_neuron_sdk_version_from_tag,
    get_neuron_release_manifest,
    is_canary_context,
    is_mainline_context,
    is_dlc_cicd_context,
    is_safety_test_context,
    run_cmd_on_container,
    start_container,
    stop_and_remove_container,
    get_repository_local_path,
    get_repository_and_tag_from_image_uri,
    get_python_version_from_image_uri,
    get_cuda_version_from_tag,
    get_labels_from_ecr_image,
    get_buildspec_path,
    get_all_the_tags_of_an_image_from_ecr,
    is_nightly_context,
    execute_env_variables_test,
    UL20_CPU_ARM64_US_WEST_2,
    UBUNTU_18_HPU_DLAMI_US_WEST_2,
    NEURON_UBUNTU_18_BASE_DLAMI_US_WEST_2,
    get_installed_python_packages_with_version,
    login_to_ecr_registry,
    get_account_id_from_image_uri,
    get_region_from_image_uri,
    DockerImagePullException,
    get_installed_python_packages_with_version,
    get_installed_python_packages_using_image_uri,
    get_image_spec_from_buildspec,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

# List of safety check vulnerability IDs to ignore. To get the ID, run safety check on the container,
# and copy paste the ID given by the safety check for a package.
# Note:- 1. This ONLY needs to be done if a package version exists that resolves this safety issue, but that version
#           cannot be used because of incompatibilities.
#        2. Ensure that IGNORE_SAFETY_IDS is always as small/empty as possible.
IGNORE_SAFETY_IDS = {
    "tensorflow": {
        "training": {
            "py2": [
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
                # for shipping pycrypto<=2.6.1 - the last available version for py2
                "35015",
            ],
            "py3": [
                # CVE vulnerabilities in TF 2.6 ignoring to be able to build TF containers
                "44715",
                "44716",
                "44717",
                "43453",
                # CVE vulnerabilities in TF < 2.7.0 ignoring to be able to build TF containers
                "42098",
                "42062",
                "41994",
                "42815",
                "42772",
                "42814",
                # tensorflow-estimator and tensorflow versions must match. For all TF versions below TF 2.9.0, we cannot upgrade tf-estimator to 2.9.0
                "48551",
                # for cryptography until we have 39.0.0 release
                "51159",
                # Keras 2.10.0 is latest, rc in place for 2.11.0+
                "51516",
            ],
        },
        "inference": {
            "py2": [
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [
                # CVE vulnerabilities in TF < 2.7.0 ignoring to be able to build TF containers
                "42098",
                "42062",
                # CVE vulnerabilities in TF 2.6 ignoring to be able to build TF containers
                "44715",
                "44716",
                "44717",
                "43453",
                # tensorflow-estimator and tensorflow versions must match. For all TF versions below TF 2.9.0, we cannot upgrade tf-estimator to 2.9.0
                "48551",
                # for cryptography until we have 39.0.0 release
                "51159",
            ],
        },
        "inference-eia": {
            "py2": [
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [
                # CVE vulnerabilities in TF < 2.7.0 ignoring to be able to build TF containers
                "42098",
                "42062",
            ],
        },
        "inference-neuron": {
            "py3": [
                # 40794, 40795, 42098, 42062, 42475: TF 1.15.5 is the last available version of TF 1
                # need to ship neuron-cc that depends tf1.15 and following cve's of tf1.15.5 is ignored
                "42098",
                "42062",
                "42475",
                "40794",
                "40795",
                "42457",
                "43748",
                "43750",
                "43747",
                "43749",
                "42446",
                "42443",
                "42450",
                "42447",
                "42464",
                "42461",
                "42466",
                "42470",
                "42473",
                "42474",
                "42472",
                "42468",
                "42460",
                "42451",
                "42456",
                "42475",
                "42469",
                "42471",
                "42465",
                "42463",
                "42462",
                "42455",
                "42453",
                "42452",
                "42459",
                "42454",
                "42449",
                "42448",
                "42444",
                "42442",
                "42445",
                "43613",
                "43453",
                "44715",
                "44717",
                "44716",
                "43750",
                "42470",
                "43749",
                "43747",
                "43748",
                "42446",
                "42447",
                "42461",
                "42471",
                "42463",
                "42452",
                "42457",
                "42454",
                "42449",
                "42444",
                "42442",
                "42443",
                "42450",
                "42464",
                "42466",
                "42473",
                "42474",
                "42472",
                "42468",
                "42456",
                "42460",
                "42451",
                "42475",
                "42469",
                "42465",
                "42462",
                "42455",
                "42453",
                "42459",
                "42448",
                "42445",
                "43613",
                "40794",
                "40795",
                "44790",
                "44788",
                "44874",
                "44763",
                "44871",
                "44870",
                "44786",
                "44852",
                "44851",
                "44848",
                "44847",
                "44845",
                "44863",
                "44867",
                "44781",
                "44865",
                "44782",
                "44860",
                "44868",
                "44789",
                "44854",
                "44853",
                "44856",
                "44794",
                "44876",
                "44778",
                "44864",
                "44866",
                "44880",
                "44784",
                "44779",
                "44777",
                "44780",
                "44862",
                "44869",
                "44783",
                "44792",
                "44787",
                "44785",
                "44858",
                "44861",
                "44857",
                "44855",
                "44796",
                "44795",
                "44793",
                "44791",
                "44859",
                "44873",
                "44850",
                "44849",
                "44846",
                "44872",
                "51499",
                "51084",
                "51049",
                "51104",
                "51081",
                "51089",
                "51051",
                "51100",
                "51069",
                "51078",
                "51101",
                "51075",
                "51065",
                "51068",
                "51093",
                "51099",
                "51092",
                "51074",
                "51056",
                "51053",
                "51064",
                "51102",
                "51083",
                "51063",
                "51094",
                "51048",
                "51054",
                "51052",
                "51055",
                "51058",
                "51090",
                "51087",
                "51095",
                "51077",
                "51105",
                "51057",
                "51047",
                "51071",
                "51097",
                "51067",
                "51059",
                "51088",
                "51103",
                "51096",
                "51086",
                "51073",
                "51066",
                "51085",
                "51079",
                "51070",
                "51082",
                "51098",
                "51061",
                "51062",
                "51072",
                "51050",
                "51076",
                "51060",
                "51948",
                "51947",
                "51945",
                "51941",
                "51962",
                "51960",
                "51961",
                "51959",
                "51953",
                "51958",
                "51956",
                "51943",
                "51954",
                "51957",
                "52348",
                "52347",
                "51955",
                "51952",
                "51951",
                "51950",
                "51949",
                "51944",
                "51963",
                "51946",
                "51516",
                "48551",
                "48639",
                "48653",
                "48641",
                "51080",
                "48640",
                "48634",
                "51091",
                "48636",
                "48649",
                "48654",
                "48638",
                "48652",
                "48648",
                "48647",
                "48651",
                "48635",
                "51167",
                "48650",
                "48643",
                "48627",
                "48642",
                "48629",
                "48644",
                "48645",
                "48637",
                "50474",
                "48646",
                "48633",
                "51095",
                "51086",
                "48645",
                "48651",
                "51073",
                "51957",
                "51071",
                "52347",
                "51093",
                "51963",
                "51962",
                "48636",
                "48637",
                "48649",
                "48639",
                "51084",
                "51068",
                "51065",
                "51080",
                "51081",
                "51103",
                "51050",
                "51958",
                "51048",
                "51959",
                "51087",
                "51101",
                "51951",
                "51092",
                "51088",
                "51054",
                "51105",
                "51083",
                "48633",
                "51069",
                "51059",
                "51052",
                "51956",
                "51948",
                "51082",
                "51090",
                "51091",
                "51946",
                "51953",
                "51075",
                "48646",
                "51085",
                "52348",
                "51097",
                "51943",
                "48642",
                "51066",
                "48634",
                "51074",
                "48648",
                "51096",
                "51954",
                "48647",
                "51098",
                "51067",
                "51056",
                "51049",
                "51100",
                "51944",
                "51952",
                "48638",
                "51064",
                "48629",
                "51941",
                "51060",
                "48653",
                "48641",
                "48652",
                "51057",
                "48654",
                "51061",
                "51960",
                "48650",
                "51070",
                "51945",
                "48643",
                "51051",
                "48551",
                "51072",
                "51949",
                "51094",
                "51950",
                "51053",
                "51063",
                "51079",
                "51077",
                "51099",
                "51104",
                "51947",
                "51078",
                "51167",
                "51058",
                "51047",
                "50474",
                "51089",
                "51055",
                "51062",
                "48640",
                "51076",
                "51961",
                "51102",
                "51955",
                "48635",
                "48644",
            ],
        },
    },
    "mxnet": {
        "inference-eia": {
            "py2": [
                # numpy<=1.16.0 -- This has to only be here while we publish MXNet 1.4.1 EI DLC v1.0
                "36810",
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [],
        },
        "inference": {
            "py2": [
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [],
        },
        "training": {
            "py2": [
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [],
        },
        "inference-neuron": {
            "py3": [
                # for shipping neuron-cc[tensorflow] that install tf1.15
                "42098",
                "42062",
                "42475",
                "40794",
                "40795",
                "42457",
                "43748",
                "43750",
                "43747",
                "43749",
                "42446",
                "42443",
                "42450",
                "42447",
                "42464",
                "42461",
                "42466",
                "42470",
                "42473",
                "42474",
                "42472",
                "42468",
                "42460",
                "42451",
                "42456",
                "42475",
                "42469",
                "42471",
                "42465",
                "42463",
                "42462",
                "42455",
                "42453",
                "42452",
                "42459",
                "42454",
                "42449",
                "42448",
                "42444",
                "42442",
                "42445",
                "43613",
                "43453",
                "44715",
                "44717",
                "44716",
                "43750",
                "42470",
                "43749",
                "43747",
                "43748",
                "42446",
                "42447",
                "42461",
                "42471",
                "42463",
                "42452",
                "42457",
                "42454",
                "42449",
                "42444",
                "42442",
                "42443",
                "42450",
                "42464",
                "42466",
                "42473",
                "42474",
                "42472",
                "42468",
                "42456",
                "42460",
                "42451",
                "42475",
                "42469",
                "42465",
                "42462",
                "42455",
                "42453",
                "42459",
                "42448",
                "42445",
                "43613",
                "40794",
                "40795",
                "44790",
                "44788",
                "44874",
                "44763",
                "44871",
                "44870",
                "44786",
                "44852",
                "44851",
                "44848",
                "44847",
                "44845",
                "44863",
                "44867",
                "44781",
                "44865",
                "44782",
                "44860",
                "44868",
                "44789",
                "44854",
                "44853",
                "44856",
                "44794",
                "44876",
                "44778",
                "44864",
                "44866",
                "44880",
                "44784",
                "44779",
                "44777",
                "44780",
                "44862",
                "44869",
                "44783",
                "44792",
                "44787",
                "44785",
                "44858",
                "44861",
                "44857",
                "44855",
                "44796",
                "44795",
                "44793",
                "44791",
                "44859",
                "44873",
                "44850",
                "44849",
                "44846",
                "44872",
                # Following are shipping neuron-cc that depends on numpy<1.20.0 (will be fixed in next release)
                "43453",
                "44715",
                "44716",
                "44717",
                # Following is for neuron-cc that depends on protobuf<=3.20.1
                "51167",
            ],
        },
    },
    "pytorch": {
        "training": {
            "py2": [
                # astropy<3.0.1
                "35810",
                # for shipping pillow<=6.2.2 - the last available version for py2
                "38449",
                "38450",
                "38451",
                "38452",
            ],
            "py3": [
                # for shipping bokeh<=2.3.3 - the last available version for py3.6
                "42772",
                "42814",
                "42815",
            ],
        },
        "training-neuron": {
            "_comment": "py2 is deprecated",
            "py2": [],
            "py3": [
                # not possible for neuron-cc
                "43453",
                "44715",
                "44717",
                "44716",
                # for releasing PT1.12 safety check tools might report a vulnerability for the package commonmarker,
                # which is a dependency of deepspeed.
                # This package is only used to build the documentation pages of deepspeed
                # and won’t be used in the package that gets installed into the DLC.
                # This security issue can be safely ignored
                # and an attempt to upgrade deepspeed version to
                # remediate it might have an inadvertent negative impact on the DLC components functionality.
                "48298",
                # for cryptography until e have 39.0.0 release
                "51159",
                # for Safety. it is test package and not part of image
                "51358",
                # Wheel is needed by tensorboard but v0.38 is not there yet
                "51499",
                # Ignored- please check https://github.com/pytest-dev/py/issues/287
                "51457",
                # Sqlalchemy latest release is not there yet
                "51668",
            ],
        },
        "inference": {
            "py3": [
                # for shipping Torchserve 0.5.2 - the last available version
                "44463",
                "44715",
                "44716",
                "44717",
                # for cryptography until e have 39.0.0 release
                "51159",
                "51358",
                # for scipy version 1.10.1 which is a hardcoded dependency of sagemaker-pytorch-inference
                "62894",
            ]
        },
        "inference-eia": {"py3": []},
        "inference-neuron": {
            "py3": [
                # 39409, 39408, 39407, 39406: TF 1.15.5 is on par with TF 2.0.4, 2.1.3, 2.2.2, 2.3.2 in security patches
                "39409",
                "39408",
                "39407",
                "39406",
                # need to ship neuron-cc that depends on tf1.15. Following are cve's in tf1.15.5
                "40794",
                "40795",
                "42098",
                "42062",
                "42475",
                "42457",
                "43748",
                "43750",
                "43747",
                "43749",
                "42446",
                "42443",
                "42450",
                "42447",
                "42464",
                "42461",
                "42466",
                "42470",
                "42473",
                "42474",
                "42472",
                "42468",
                "42460",
                "42451",
                "42456",
                "42475",
                "42469",
                "42471",
                "42465",
                "42463",
                "42462",
                "42455",
                "42453",
                "42452",
                "42459",
                "42454",
                "42449",
                "42448",
                "42444",
                "42442",
                "42445",
                "43613",
                "43453",
                "44715",
                "44717",
                "44716",
                "43750",
                "42470",
                "43749",
                "43747",
                "43748",
                "42446",
                "42447",
                "42461",
                "42471",
                "42463",
                "42452",
                "42457",
                "42454",
                "42449",
                "42444",
                "42442",
                "42443",
                "42450",
                "42464",
                "42466",
                "42473",
                "42474",
                "42472",
                "42468",
                "42456",
                "42460",
                "42451",
                "42475",
                "42469",
                "42465",
                "42462",
                "42455",
                "42453",
                "42459",
                "42448",
                "42445",
                "43613",
                "40794",
                "40795",
                "44790",
                "44788",
                "44874",
                "44763",
                "44871",
                "44870",
                "44786",
                "44852",
                "44851",
                "44848",
                "44847",
                "44845",
                "44863",
                "44867",
                "44781",
                "44865",
                "44782",
                "44860",
                "44868",
                "44789",
                "44854",
                "44853",
                "44856",
                "44794",
                "44876",
                "44778",
                "44864",
                "44866",
                "44880",
                "44784",
                "44779",
                "44777",
                "44780",
                "44862",
                "44869",
                "44783",
                "44792",
                "44787",
                "44785",
                "44858",
                "44861",
                "44857",
                "44855",
                "44796",
                "44795",
                "44793",
                "44791",
                "44859",
                "44873",
                "44850",
                "44849",
                "44846",
                "44872",
                # for shipping Torchserve 0.5.2 - the last available version
                "44463",
                # protobuf.. neuron-cc depends on this
                "51167",
                "51084",
                "51049",
                "51104",
                "51081",
                "51089",
                "51051",
                "51100",
                "51069",
                "51078",
                "51101",
                "51075",
                "51065",
                "51068",
                "51093",
                "51099",
                "51092",
                "51074",
                "51056",
                "51053",
                "51064",
                "51102",
                "51083",
                "51063",
                "51094",
                "51048",
                "51080",
                "51054",
                "51052",
                "51055",
                "51058",
                "51090",
                "51087",
                "51095",
                "51077",
                "51105",
                "51057",
                "51047",
                "51071",
                "51097",
                "51091",
                "51067",
                "51059",
                "51088",
                "51103",
                "51096",
                "51086",
                "51073",
                "51066",
                "51085",
                "51079",
                "51070",
                "51082",
                "51098",
                "51061",
                "51062",
                "51072",
                "51050",
                "51076",
                "51060",
                "51948",
                "51947",
                "51941",
                "51945",
                "51962",
                "51960",
                "51961",
                "51959",
                "51953",
                "51958",
                "51956",
                "51943",
                "51954",
                "51957",
                "52348",
                "52347",
                "51955",
                "51952",
                "51951",
                "51950",
                "51949",
                "51944",
                "51963",
                "51946",
                "48654",
                "48641",
                "48629",
                "48635",
                "48651",
                "48652",
                "48638",
                "48645",
                "48646",
                "48633",
                "48643",
                "48634",
                "48640",
                "48648",
                "48639",
                "48644",
                "48636",
                "48653",
                "48649",
                "48642",
                "48637",
                "48647",
                "48650",
                "48551",
            ]
        },
    },
    "autogluon": {
        "training": {
            "py3": [
                # Pydantic 1.10.2 prevents long strings as int inputs to fix CVE-2020-10735 - upstream dependencies are still not patched
                "50916",
                # Safety 2.2.0 updates its dependency 'dparse' to include a security fix. - not packaged with container, result of security scanning process
                "51358",
            ]
        },
        "inference": {
            "py3": [
                # Pydantic 1.10.2 prevents long strings as int inputs to fix CVE-2020-10735 - upstream dependencies are still not patched
                "50916",
                # Safety 2.2.0 updates its dependency 'dparse' to include a security fix. - not packaged with container, result of security scanning process
                "51358",
            ]
        },
    },
}


def _get_safety_ignore_list(image_uri):
    """
    Get a list of known safety check issue IDs to ignore, if specified in IGNORE_LISTS.
    :param image_uri:
    :return: <list> list of safety check IDs to ignore
    """
    if "mxnet" in image_uri:
        framework = "mxnet"
    elif "pytorch" in image_uri:
        framework = "pytorch"
    elif "autogluon" in image_uri:
        framework = "autogluon"
    else:
        framework = "tensorflow"

    job_type = (
        "training-neuronx"
        if "training-neuronx" in image_uri
        else "training-neuron"
        if "training-neuron" in image_uri
        else "training"
        if "training" in image_uri
        else "inference-eia"
        if "eia" in image_uri
        else "inference-neuronx"
        if "inference-neuronx" in image_uri
        else "inference-neuron"
        if "inference-neuron" in image_uri
        else "inference"
    )
    python_version = "py2" if "py2" in image_uri else "py3"

    return IGNORE_SAFETY_IDS.get(framework, {}).get(job_type, {}).get(python_version, [])


def _get_latest_package_version(package):
    """
    Get the latest package version available on pypi for a package.
    It is retried multiple times in case there are transient failures in executing the command.

    :param package: str Name of the package whose latest version must be retrieved
    :return: tuple(command_success: bool, latest_version_value: str)
    """
    # safe
    pypi_package_info = requests.get(f"https://pypi.org/pypi/{package}/json")
    data = json.loads(pypi_package_info.text)
    versions = data["releases"].keys()
    return str(max(Version(v) for v in versions))


@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
@pytest.mark.canary("Run safety tests regularly on production images")
@pytest.mark.skipif(
    not is_dlc_cicd_context(), reason="Skipping test because it is not running in dlc cicd infra"
)
@pytest.mark.skipif(
    not (is_safety_test_context()),
    reason=(
        "Skipping the test to decrease the number of calls to the Safety Check DB. "
        "Test will be executed in the 'mainline' pipeline and canaries pipeline."
    ),
)
def test_safety(image):
    """
    Runs safety check on a container with the capability to ignore safety issues that cannot be fixed, and only raise
    error if an issue is fixable.
    """
    from dlc.safety_check import SafetyCheck

    safety_check = SafetyCheck()

    repo_name, image_tag = image.split("/")[-1].split(":")
    ignore_ids_list = _get_safety_ignore_list(image)
    sep = " -i "
    ignore_str = "" if not ignore_ids_list else f"{sep}{sep.join(ignore_ids_list)}"

    container_name = f"{repo_name}-{image_tag}-safety"
    docker_exec_cmd = f"docker exec -i {container_name}"
    test_file_path = os.path.join(CONTAINER_TESTS_PREFIX, "testSafety")

    # Add null entrypoint to ensure command exits immediately
    run(
        f"docker run -id "
        f"--name {container_name} "
        f"--mount type=bind,src=$(pwd)/container_tests,target=/test "
        f"--entrypoint='/bin/bash' "
        f"{image}",
        hide=True,
    )
    try:
        run(f"{docker_exec_cmd} pip install 'safety>=2.2.0' yolk3k ", hide=True)
        json_str_safety_result = safety_check.run_safety_check_on_container(docker_exec_cmd)
        safety_result = json.loads(json_str_safety_result)["vulnerabilities"]
        for vulnerability in safety_result:
            package = vulnerability["package_name"]
            affected_versions = vulnerability["vulnerable_spec"]
            vulnerability_id = vulnerability["vulnerability_id"]

            # Get the latest version of the package with vulnerability
            latest_version = _get_latest_package_version(package)
            # If the latest version of the package is also affected, igvnore this vulnerability
            if Version(latest_version) in SpecifierSet(affected_versions):
                # Version(x) gives an object that can be easily compared with another version, or with a SpecifierSet.
                # Comparing two versions as a string has some edge cases which require us to write more code.
                # SpecifierSet(x) takes a version constraint, such as "<=4.5.6", ">1.2.3", or ">=1.2,<3.4.5", and
                # gives an object that can be easily compared against a Version object.
                # https://packaging.pypa.io/en/latest/specifiers/
                ignore_str += f" -i {vulnerability_id}"
        assert (
            safety_check.run_safety_check_with_ignore_list(docker_exec_cmd, ignore_str) == 0
        ), f"Safety test failed for {image}"
    finally:
        run(f"docker rm -f {container_name}", hide=True)

@pytest.mark.usefixtures("sagemaker")
@pytest.mark.integration("oss_compliance")
@pytest.mark.model("N/A")
@pytest.mark.skipif(
    not is_dlc_cicd_context(), reason="We need to test OSS compliance only on PRs and pipelines"
)
def test_oss_compliance(image):
    """
    Run oss compliance check on a container to check if license attribution files exist.
    And upload source of third party packages to S3 bucket.
    """
    THIRD_PARTY_SOURCE_CODE_BUCKET = "aws-dlinfra-licenses"
    THIRD_PARTY_SOURCE_CODE_BUCKET_PATH = "third_party_source_code"
    file = "THIRD_PARTY_SOURCE_CODE_URLS"
    container_name = get_container_name("oss_compliance", image)
    context = Context()
    local_repo_path = get_repository_local_path()
    start_container(container_name, image, context)

    # run compliance test to make sure license attribution files exists. testOSSCompliance is copied as part of Dockerfile
    run_cmd_on_container(container_name, context, "/usr/local/bin/testOSSCompliance /root")

    try:
        context.run(
            f"docker cp {container_name}:/root/{file} {os.path.join(local_repo_path, file)}"
        )
    finally:
        context.run(f"docker rm -f {container_name}", hide=True)

    s3_resource = boto3.resource("s3")

    with open(os.path.join(local_repo_path, file)) as source_code_file:
        for line in source_code_file:
            name, version, url = line.split(" ")
            file_name = f"{name}_v{version}_source_code"
            s3_object_path = f"{THIRD_PARTY_SOURCE_CODE_BUCKET_PATH}/{file_name}.tar.gz"
            local_file_path = os.path.join(local_repo_path, file_name)

            for i in range(3):
                try:
                    if not os.path.isdir(local_file_path):
                        context.run(f"git clone {url.rstrip()} {local_file_path}", hide=True)
                        context.run(f"tar -czvf {local_file_path}.tar.gz {local_file_path}")
                except Exception as e:
                    time.sleep(1)
                    if i == 2:
                        LOGGER.error(f"Unable to clone git repo. Error: {e}")
                        raise
                    continue
            try:
                if os.path.exists(f"{local_file_path}.tar.gz"):
                    LOGGER.info(f"Uploading package to s3 bucket: {line}")
                    s3_resource.Object(THIRD_PARTY_SOURCE_CODE_BUCKET, s3_object_path).load()
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    try:
                        # using aws cli as using boto3 expects to upload folder by iterating through each file instead of entire folder.
                        context.run(
                            f"aws s3 cp {local_file_path}.tar.gz s3://{THIRD_PARTY_SOURCE_CODE_BUCKET}/{s3_object_path}"
                        )
                        object = s3_resource.Bucket(THIRD_PARTY_SOURCE_CODE_BUCKET).Object(
                            s3_object_path
                        )
                        object.Acl().put(ACL="public-read")
                    except ClientError as e:
                        LOGGER.error(
                            f"Unable to upload source code to bucket {THIRD_PARTY_SOURCE_CODE_BUCKET}. Error: {e}"
                        )
                        raise
                else:
                    LOGGER.error(
                        f"Unable to check if source code is present on bucket {THIRD_PARTY_SOURCE_CODE_BUCKET}. Error: {e}"
                    )
                    raise

@pytest.mark.usefixtures("sagemaker")
@pytest.mark.model("N/A")
def test_core_package_version(image):
    """
    In this test, we ensure that if a core_packages.json file exists for an image, the packages installed in the image
    satisfy the version constraints specified in the core_packages.json file.
    """
    core_packages_path = src_utils.get_core_packages_path(image)
    if not os.path.exists(core_packages_path):
        pytest.skip(f"Core packages file {core_packages_path} does not exist for {image}")
    LOGGER.info(f"Core packages file {core_packages_path} for {image}")

    with open(core_packages_path, "r") as f:
        core_packages = json.load(f)

    ctx = Context()
    container_name = get_container_name("test_core_package_version", image)
    start_container(container_name, image, ctx)
    docker_exec_command = f"""docker exec --user root {container_name}"""
    installed_package_version_dict = get_installed_python_packages_with_version(docker_exec_command)

    violation_data = {}

    for package_name, specs in core_packages.items():
        package_name = package_name.lower()
        installed_version = None
        if package_name not in installed_package_version_dict:
            violation_data[
                package_name
            ] = f"Package: {package_name} not installed in {installed_package_version_dict}"
        else:
            installed_version = Version(installed_package_version_dict[package_name])
        if installed_version and installed_version not in SpecifierSet(
            specs.get("version_specifier")
        ):
            violation_data[package_name] = (
                f"Package: {package_name} version {installed_version} does not match "
                f"requirement {specs.get('version_specifier')}"
            )

    stop_and_remove_container(container_name, ctx)
    assert (
        not violation_data
    ), f"Few packages violate the core_package specifications: {violation_data}"


@pytest.mark.model("N/A")
def test_package_version_regression_in_image(image):
    """
    This test verifies if the python package versions in the already released image are not being downgraded/deleted in the
    new released. This test would be skipped for images whose BuildSpec does not have `latest_release_tag` or `release_repository`
    keys in the buildspec - as these keys are used to extract the released image uri. Additionally, if the image is not already
    released, this test would be skipped.
    """
    dlc_path = os.getcwd().split("/test/")[0]
    corresponding_image_spec = get_image_spec_from_buildspec(
        image_uri=image, dlc_folder_path=dlc_path
    )

    if any(
        [
            expected_key not in corresponding_image_spec
            for expected_key in ["latest_release_tag", "release_repository"]
        ]
    ):
        pytest.skip(f"Image {image} does not have `latest_release_tag` in its buildspec.")

    previous_released_image_uri = f"""{corresponding_image_spec["release_repository"]}:{corresponding_image_spec["latest_release_tag"]}"""

    LOGGER.info(f"Image spec for {image}: {json.dumps(corresponding_image_spec)}")
    ctx = Context()

    # Pull previous_released_image_uri
    prod_account_id = get_account_id_from_image_uri(image_uri=previous_released_image_uri)
    prod_image_region = get_region_from_image_uri(image_uri=previous_released_image_uri)
    login_to_ecr_registry(context=ctx, account_id=prod_account_id, region=prod_image_region)
    previous_released_image_uri = f"{previous_released_image_uri}"
    run_output = ctx.run(f"docker pull {previous_released_image_uri}", warn=True, hide=True)
    if not run_output.ok:
        if "requested image not found" in run_output.stderr.lower():
            pytest.skip(
                f"Previous image: {previous_released_image_uri} for Image: {image} has not been released yet"
            )
        raise DockerImagePullException(
            f"{run_output.stderr} when pulling {previous_released_image_uri}"
        )

    # Get the installed python package versions and find any regressions
    current_image_package_version_dict = get_installed_python_packages_using_image_uri(
        context=ctx, image_uri=image
    )
    released_image_package_version_dict = get_installed_python_packages_using_image_uri(
        context=ctx, image_uri=previous_released_image_uri
    )
    violating_packages = {}
    for package_name, version_in_released_image in released_image_package_version_dict.items():
        if package_name not in current_image_package_version_dict:
            violating_packages[
                package_name
            ] = "Not present in the image that is being currently built."
            continue
        version_in_current_image = current_image_package_version_dict[package_name]
        if Version(version_in_released_image) > Version(version_in_current_image):
            violating_packages[
                package_name
            ] = f"Version in already released image: {version_in_released_image} is greater that version in current image: {version_in_current_image}"

    assert (
        not violating_packages
