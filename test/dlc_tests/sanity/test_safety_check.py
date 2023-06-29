import json
import logging
import os
import sys

from packaging.specifiers import SpecifierSet
from packaging.version import Version

import pytest
import requests

from invoke import run

from test.test_utils import (
    CONTAINER_TESTS_PREFIX,
    is_dlc_cicd_context,
    is_canary_context,
    is_mainline_context,
    is_safety_test_context,
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
                # Fix for issue available in torch version 1.13.1
                "54718",
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
                # Fix for issue available in torch version 1.13.1
                "54718",
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
                # Fix for issue available in torch version 1.13.1
                "54718",
            ]
        },
        "inference-eia": {
            "py3": [ # Fix for issue available in torch version 1.13.1
            "54718"
            ]
        },
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
                # Fix for issue available in torch version 1.13.1
                "54718",
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
        "training-neuron"
        if "training-neuron" in image_uri
        else "training"
        if "training" in image_uri
        else "inference-eia"
        if "eia" in image_uri
        else "inference-neuron"
        if "inference-neuron" in image_uri
        else "inference"
    )
    python_version = "py2" if "py2" in image_uri else "py3"
    print("SHOW" + IGNORE_SAFETY_IDS.get(framework, {}).get(job_type, {}).get(python_version, []))

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
