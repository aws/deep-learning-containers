import os
import re
import git
import sys
import logging
from packaging.version import Version #, parse

from invoke.context import Context

PUBLIC_DLC_REGISTRY = "763104351884"
PUBLIC_DLC_REGION = "us-west-2"

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

def get_sagemaker_images_from_github(registry, framework, region, image_type, processor=None):
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

    framework_versions = versions if len(versions) < 4 else versions[:3]
    image_definitions = []
    for fw_version in framework_versions:
        if fw_version in pre_populated_py_version:
            py_versions = pre_populated_py_version[fw_version]
        else:
            py_versions = [get_canary_default_tag_py3_version(framework, fw_version)]
        for py_version in py_versions:
            image_definition = []
            if processor == "gpu" or processor is None:
                image_definitions.append({
                            "registry": registry,
                            "region": region,
                            "repository": f"{framework}-{image_type}",
                            "framework_version": fw_version,
                            "processor": processor,
                            "py_version": py_version
                        })
            if processor == "cpu" or processor is None:
                image_definitions.append({
                            "registry": registry,
                            "region": region,
                            "repository": f"{framework}-{image_type}",
                            "framework_version": fw_version,
                            "processor": processor,
                            "py_version": py_version
                        })

    return image_definitions


def run_endpoint_tests(account_id, sagemaker_region, registry, region, repository, framework_name, framework_version, python_version, tag):
    env_variables = {}
    retries = "--reruns 2"
    instance_type = "p3.8xlarge"
    python_version = ("2" if "27" in python_version else
                      "3" if "36" in python_version else
                      python_version.lstrip("py"))

    test_location = os.path.join("test", "sagemaker_endpoint_tests", framework_name)

    ctx = Context()
    with ctx.cd(test_location):
        run_out = ctx.run(f"pytest -rs {retries} test_endpoint.py "
            f"--account-id {account_id} "
            f"--region {region} "
            f"--registry {registry} "
            f"--repository {repository} "
            f"--instance-type ml.{instance_type} "
            f"--framework-version {framework_version} "
            f"--sagemaker-region {sagemaker_region} "
            f"--py-version {python_version} "
            f"--tag {tag}",
            warn=True, env=env_variables, echo=True
        )
    return run_out.ok, run_out.stdout

def main():
    frameworks = [os.environ["FRAMEWORK"]] # ["mxnet","pytorch", "tensorflow"]
    canary_account_id = os.environ["ACCOUNT_ID"]
    sagemaker_region = os.environ["REGION"]
    image_type = os.environ["IMAGE_TYPE"]
    registry = os.getenv("PUBLIC_DLC_REGISTRY") or PUBLIC_DLC_REGISTRY
    region = os.getenv("PUBLIC_DLC_REGION") or PUBLIC_DLC_REGION
    
    # assuming that the regions for sagemaker and image are the same
    for framework in frameworks:
        gpu_image_definitions = get_sagemaker_images_from_github(registry, framework, region, image_type, processor="gpu")
        for image_definition in gpu_image_definitions:
            repository = image_definition["repository"]
            framework_version = image_definition["framework_version"]
            processor = image_definition["processor"]
            py_version = image_definition["py_version"]
            domain_suffix = ".cn" if region in ("cn-north-1", "cn-northwest-1") else ""
            image_uri = f"{registry}.dkr.ecr.{region}.amazonaws.com{domain_suffix}/{repository}:{framework_version}-{processor}-{py_version}"
            test_status, test_logs = run_endpoint_tests(canary_account_id, sagemaker_region, registry, region, repository, framework, framework_version, py_version, f"{framework_version}-{processor}-{py_version}")
            if not test_status:
                LOGGER.error(f"Endpoint test failed for image {image_uri}. {test_logs}")

if __name__ == "__main__":
    main()
