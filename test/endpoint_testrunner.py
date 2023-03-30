import os
import re
import git
import sys
import logging
from packaging.version import Version #, parse

from invoke.context import Context

PUBLIC_DLC_REGISTRY = "763104351884"


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


def get_canary_default_tag_py3_version(framework, version):
    """
    Currently, only TF2.2 images and above have major/minor python version in their canary tag. Creating this function
    to conditionally choose a python version based on framework version ranges. If we move up to py38, for example,
    this is the place to make the conditional change.
    :param framework: tensorflow1, tensorflow2, mxnet, pytorch
    :param version: fw major.minor version, i.e. 2.2
    :return: default tag python version
    """
    if framework == "tensorflow":
        if Version("2.2") <= Version(version) < Version("2.6"):
            return "py37"
        if Version("2.6") <= Version(version) < Version("2.8"):
            return "py38"
        if Version(version) >= Version("2.8"):
            return "py39"

    if framework == "mxnet":
        if Version(version) == Version("1.8"):
            return "py37"
        if Version(version) >= Version("1.9"):
            return "py38"

    if framework == "pytorch":
        if Version("1.9") <= Version(version) < Version("1.13"):
            return "py38"
        if Version(version) >= Version("1.13") and Version(version) < Version("2.0"):
            return "py39"
        if Version(version) >= Version("2.0"):
            return "py310"

    return "py3"

def get_sagemaker_images_from_github(framework, region, image_type):
    """
    Return which Inference GPU canary images to run canary tests on for a given framework and AWS region

    :param framework: ML framework (mxnet, tensorflow, pytorch)
    :param region: AWS region
    :param image_type: inference
    :return: a list of space separated image uris
    """
    customer_type_tag = "-sagemaker"
    version_regex = {
        "tensorflow": rf"tf(-sagemaker)?{customer_type_tag}-(\d+.\d+)",
        "mxnet": rf"mx(-sagemaker)?{customer_type_tag}-(\d+.\d+)",
        "pytorch": rf"pt(-sagemaker)?{customer_type_tag}-(\d+.\d+)",
    }

    # Get tags from repo releases
    repo = git.Repo(os.getcwd(), search_parent_directories=True)

    versions_counter = {}
    pre_populated_py_version = {}

    for tag in repo.tags:
        tag_str = str(tag)
        
        match = re.search(version_regex[framework], tag_str)
        ## The tags not having -py3 will not pass the condition below
        ## This eliminates all the old and testing tags that we are not monitoring.
        if match:
            version = match.group(2)
            if not versions_counter.get(version):
                versions_counter[version] = {"tr": False, "inf": False}

            if "tr" not in tag_str and "inf" not in tag_str:
                versions_counter[version]["tr"] = True
                versions_counter[version]["inf"] = True
            elif "tr" in tag_str:
                versions_counter[version]["tr"] = True
            elif "inf" in tag_str:
                versions_counter[version]["inf"] = True

            try:
                python_version_extracted_through_regex = match.group(3)
                if python_version_extracted_through_regex:
                    if version not in pre_populated_py_version:
                        pre_populated_py_version[version] = set()
                    pre_populated_py_version[version].add(python_version_extracted_through_regex)
            except IndexError:
                LOGGER.info(f"For Framework: {framework} we do not use regex to fetch python version")

    versions = []
    for version, inf_train in versions_counter.items():
        if inf_train["inf"]:
            versions.append(version)

    # Sort ascending to descending, use lambda to ensure 2.2 < 2.15, for instance
    versions.sort(key=lambda version_str: [int(point) for point in version_str.split(".")], reverse=True)

    registry = PUBLIC_DLC_REGISTRY
    framework_versions = versions if len(versions) < 4 else versions[:3]
    dlc_images = []
    for fw_version in framework_versions:
        if fw_version in pre_populated_py_version:
            py_versions = pre_populated_py_version[fw_version]
        else:
            py_versions = [get_canary_default_tag_py3_version(framework, fw_version)]
        for py_version in py_versions:
            images = {
                "tensorflow": {
                    "training": [
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/tensorflow-training:{fw_version}-gpu-{py_version}",
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/tensorflow-training:{fw_version}-cpu-{py_version}",
                    ],
                    "inference": [
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/tensorflow-inference:{fw_version}-gpu",
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/tensorflow-inference:{fw_version}-cpu",
                    ]
                },
                "mxnet": {
                    "training": [
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/mxnet-training:{fw_version}-gpu-{py_version}",
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/mxnet-training:{fw_version}-cpu-{py_version}",
                    ],
                    "inference": [
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/mxnet-inference:{fw_version}-gpu-{py_version}",
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/mxnet-inference:{fw_version}-cpu-{py_version}",
                    ],
                },
                "pytorch": {
                    "training": [
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/pytorch-training:{fw_version}-gpu-{py_version}",
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/pytorch-training:{fw_version}-cpu-{py_version}",
                    ],
                    "inference": [
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/pytorch-inference:{fw_version}-gpu-{py_version}",
                        f"{registry}.dkr.ecr.{region}.amazonaws.com/pytorch-inference:{fw_version}-cpu-{py_version}",
                    ],
                },
            }
            dlc_images += images[framework][image_type]

    dlc_images.sort()
    return " ".join(dlc_images)


def run_sagemaker_gpu_endpoint_tests(account_id, region, repository, framework_name, framework_version, tag):
    env_variables = {}
    retries = "--reruns 2"
    instance_type = "p3.8xlarge"
    python_version = ("2" if "27" in python_version else
                      "3" if "36" in python_version else
                      python_version.lstrip("py"))

    test_location = os.path.join("test", "sagemaker_tests", framework_name, "inference")
    if framework_name == "tensorflow":
        test_location = os.path.join(test_location, "test")

    ctx = Context()
    with ctx.cd(test_location):
        if framework_name == "mxnet":
            run_out = ctx.run(f"pytest {retries} "
                "integration/sagemaker/test_hosting.py::test_hosting "
                f"--aws-id {account_id} "
                f"--region {region} "
                f"--docker-base-name {repository} "
                f"--instance-type ml.{instance_type} "
                f"--framework-version {framework_version} "
                f"--processor gpu "
                f"--sagemaker-regions {region} "
                f"--py-version {python_version} "
                f"{tag}",
                warn=True, env=env_variables, echo=True)

        if framework_name == "pytorch":
            run_out = ctx.run(f"python -m pytest {retries} "
                "integration/sagemaker/test_mnist.py "
                f"--aws-id {account_id} "
                f"--region {region} "
                f"--docker-base-name {repository} "
                f"--instance-type ml.{instance_type} "
                f"--framework-version {framework_version} "
                f"--processor gpu "
                f"--sagemaker-region {region} "
                f"--py-version {python_version} "
                f"{image_tag_arg}",
                warn=True, env=env_variables, echo=True)

        if framework_name == "tensorflow":
            run_out = ctx.run(f"python -m pytest {retries} "
                "integration/sagemaker/test_tfs.py::test_tfs_model "
                f"--registry {account_id} "
                f"--region {region} "
                f"--repo {repository} "
                f"--instance-types ml.{instance_type} "
                f"--sagemaker-regions {region} "
                f"--versions {framework_version} "
                f"{tag}",
                warn=True, env=env_variables, echo=True)

    return run_out.ok, run_out.stdout

# sagemaker endpoints are for inference purposes only
def run_sagemaker_endpoint_tests(account_id, region, repository, framework_name, framework_version, processor, tag):
    if processor == "gpu":
        run_sagemaker_gpu_endpoint_tests(
            account_id, region, repository, framework_name, framework_version, tag
        )
        

if __name__ == "__main__":
    framework = "mxnet"
    region = "us-west-2"
    image_type = "training"
    result = get_sagemaker_images_from_github(framework, region, image_type)
    gpu_images = [image for image in result.split(" ") if "gpu" in image]
    print(gpu_images)
