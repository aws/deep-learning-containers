import os
import re
import git
import sys
import logging
from multiprocessing import Pool
from packaging.version import Version #, parse
from junit_xml import TestSuite, TestCase
from invoke.context import Context
from sagemaker_endpoint_tests import utils

PUBLIC_DLC_REGISTRY = "763104351884"
PUBLIC_DLC_REGION = "us-west-2"
DEFAULT_REPORT_XML = os.path.join(os.getcwd(),"test", "default_report.xml")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

def generate_report(report, test_name, test_class, test_out, test_err=''):
    """
    Generate junitxml test report
    :param report: CodeBuild Report
    Returns: None
    """
    # TestCase(name, classname, test time, system-out, system-err)
    test_cases = [TestCase(test_name, test_class, 1, test_out, test_err)]
    ts = TestSuite(report, test_cases)
    with open(report, "w") as report_file:
        TestSuite.to_file(report_file, [ts], prettyprint=False)


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


def get_sagemaker_images(images_str, processor):
    """
    Gets sagemaker images from images string
    Assume that all image are of the same framework
    :param images_str: string containing image uris
    :param processor: string representing processor type
    :return: a tuple - framework and a list of image definitions
    """
    framework = None 
    image_definitions = []
    image_uris = images_str.split(" ")
    for image_uri in image_uris:
        if processor in image_uri:
            if framework is None:
                framework = utils.get_framework_name(image_uri)
            py_version = utils.get_py_version(image_uri)
            framework = utils.get_framework_name(image_uri)
            framework_version = utils.get_framework_version(image_uri)
            repository, tag = utils.get_repository_and_tag_from_image_uri(image_uri)
            registry = utils.get_account_id_from_image_uri(image_uri)
            region = utils.get_region_from_image_uri(image_uri)

            image_definitions.append({
                "registry": registry,
                "region": region,
                "repository": repository,
                "framework_version": framework_version,
                "processor": processor,
                "py_version": py_version,
                "tag": tag
            })

    return framework, image_definitions


def get_sagemaker_images_from_github(registry, framework, region, image_type, processor=None):
    """
    Return which Inference GPU canary images to run canary tests on for a given framework and AWS region

    :param framework: ML framework (mxnet, tensorflow, pytorch)
    :param region: AWS region
    :param image_type: inference
    :return: a list of space separated image uris
    """
    
    version_regex = {
        "tensorflow": rf"tf(-sagemaker)-(\d+.\d+)",
        "mxnet": rf"mx(-sagemaker)-(\d+.\d+)",
        "pytorch": rf"pt(-sagemaker)-(\d+.\d+)",
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

    versions.sort(key=lambda version_str: [int(point) for point in version_str.split(".")], reverse=True)

    framework_versions = versions if len(versions) < 4 else versions[:3]
    image_definitions = []
    for fw_version in framework_versions:
        if fw_version in pre_populated_py_version:
            py_versions = pre_populated_py_version[fw_version]
        else:
            py_versions = [get_canary_default_tag_py3_version(framework, fw_version)]
        for py_version in py_versions:
            if processor == "gpu" or processor is None:
                image_definitions.append({
                            "registry": registry,
                            "region": region,
                            "repository": f"{framework}-{image_type}",
                            "framework_version": fw_version,
                            "processor": "gpu",
                            "py_version": py_version,
                            "tag": f"{fw_version}-gpu-{py_version}"
                        })
            if processor == "cpu" or processor is None:
                image_definitions.append({
                            "registry": registry,
                            "region": region,
                            "repository": f"{framework}-{image_type}",
                            "framework_version": fw_version,
                            "processor": "cpu",
                            "py_version": py_version,
                            "tag": f"{fw_version}-cpu-{py_version}"
                        })

    return image_definitions


def execute_endpoint_test(framework_name, image_definition, account_id, sagemaker_region, registry, region):
    
    repository = image_definition.get("repository")
    framework_version = image_definition.get("framework_version")
    processor = image_definition.get("processor")
    python_version = image_definition.get("py_version")
    tag = image_definition.get("tag")

    retries = "" # " --reruns 2"
    instance_type = "g4dn.xlarge"
    python_version = "3" if "36" in python_version else python_version.lstrip("py")

    #test_location = os.path.join("test", "sagemaker_endpoint_tests", framework_name)
    test_location = os.path.join("test", "sagemaker_endpoint_tests")

    ctx = Context()
    with ctx.cd(test_location):
        #run_out = ctx.run(f"pytest -rs{retries} --junitxml=result.xml test_endpoint.py "
        run_out = ctx.run(f"pytest -rs{retries} -F {framework_name} --junitxml=result.xml test_endpoint.py "
            f"--account-id {account_id} "
            f"--region {region} "
            f"--registry {registry} "
            f"--repository {repository} "
            f"--instance-type ml.{instance_type} "
            f"--framework-version {framework_version} "
            f"--sagemaker-region {sagemaker_region} "
            f"--py-version {python_version} "
            f"--tag {tag}",
            warn=True, echo=True
        )
    return run_out.ok, run_out.stdout


def generate_image_uri(image, registry, region):
    repository = image.get("repository")
    framework_version = image.get("framework_version")
    processor = image.get("processor")
    py_version = image.get("py_version")
    domain_suffix = ".cn" if region in ("cn-north-1", "cn-northwest-1") else ""
    return f"{registry}.dkr.ecr.{region}.amazonaws.com{domain_suffix}/{repository}:{framework_version}-{processor}-{py_version}"


def run_framework_tests(framework, images, canary_account_id, sagemaker_region, registry, region):
    if not images:
        generate_report(DEFAULT_REPORT_XML, "{framework} endpoint test", "sagemaker-endpoint", "No framework iaage to test.")
        return
    
    """
    # enable if there are sufficient instances in region
    # to run process in parallel
    results = []
    pool_number = len(images)
    with Pool(pool_number) as p:
        results = p.starmap(
            execute_endpoint_test,
            [[framework, images[i], canary_account_id, sagemaker_region, registry, region] for i in range(pool_number)]
        )
    """
    # run tests sequentially
    for image in images:
        test_status, test_logs = execute_endpoint_test(framework, image, canary_account_id, sagemaker_region, registry, region)
        if not test_status:
            image_uri = generate_image_uri(image, registry, region)
            LOGGER.error(f"Endpoint test failed for image {image_uri}")


def main():

    processor = "gpu"
    image_definitions = []
    images_str = os.getenv("DLC_IMAGES")
    framework = os.getenv("FRAMEWORK") # e.g. ["mxnet","pytorch","tensorflow"]
    image_type = os.getenv("IMAGE_TYPE")
    sagemaker_region = os.getenv("REGION")
    canary_account_id = os.getenv("ACCOUNT_ID")
    region = os.getenv("PUBLIC_DLC_REGION") or PUBLIC_DLC_REGION
    registry = os.getenv("PUBLIC_DLC_REGISTRY") or PUBLIC_DLC_REGISTRY
    
    if framework:
        image_definitions = get_sagemaker_images_from_github(registry, framework, region, image_type, processor=processor)
    elif images_str:
        framework, image_definitions = get_sagemaker_images(images_str, processor)
    
    if image_definitions:
        run_framework_tests(framework, image_definitions, canary_account_id, sagemaker_region, registry, region)
    else:
        generate_report(DEFAULT_REPORT_XML, "Endpoint test", "sagemaker-endpoint", "Framework name missing or not provided.")

if __name__ == "__main__":
    main()
