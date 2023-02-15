import re
from enum import IntEnum
import subprocess
import os
from miscellaneous import *
from time import sleep, time
from base64 import b64decode
import logging
import sys
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))
from packaging.specifiers import SpecifierSet
from packaging.version import Version, parse
from glob import glob

ECR_ENHANCED_SCANNING_REPO_NAME = "ecr-enhanced-scanning-dlc-repo"
ECR_ENHANCED_REPO_REGION = "us-west-1"

def get_account_id_from_image_uri(image_uri):
    """
    Find the account ID where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Account ID
    """
    return image_uri.split(".")[0]

def get_ecr_login_boto3(ecr_client, account_id, region):
    """
    Get ECR login using boto3
    """
    user_name, password = None, None
    result = ecr_client.get_authorization_token()
    for auth in result["authorizationData"]:
        auth_token = b64decode(auth["authorizationToken"]).decode()
        user_name, password = auth_token.split(":")
    return user_name, password

def delete_file(file_path):
    subprocess.check_output(f"rm -rf {file_path}", shell=True, executable="/bin/bash")

def save_credentials_to_file(file_path, password):
    with open(file_path, "w") as file:
        file.write(f"{password}")

def get_region_from_image_uri(image_uri):
    """
    Find the region where the image is located

    :param image_uri: <str> ECR image URI
    :return: <str> AWS Region Name
    """
    region_pattern = r"(us(-gov)?|ap|ca|cn|eu|sa)-(central|(north|south)?(east|west)?)-\d+"
    region_search = re.search(region_pattern, image_uri)
    assert region_search, f"{image_uri} must have region that matches {region_pattern}"
    return region_search.group()

def ecr_repo_exists(ecr_client, repo_name, account_id=None):
    """
    :param ecr_client: boto3.Client for ECR
    :param repo_name: str ECR Repository Name
    :param account_id: str Account ID where repo is expected to exist
    :return: bool True if repo exists, False if not
    """
    query = {"repositoryNames": [repo_name]}
    if account_id:
        query["registryId"] = account_id
    try:
        ecr_client.describe_repositories(**query)
    except ecr_client.exceptions.RepositoryNotFoundException as e:
        return False
    return True

def get_repository_and_tag_from_image_uri(image_uri):
    """
    Return the name of the repository holding the image

    :param image_uri: URI of the image
    :return: <str> repository name
    """
    repository_uri, tag = image_uri.split(":")
    _, repository_name = repository_uri.split("/")
    return repository_name, tag

def get_all_the_tags_of_an_image_from_ecr(ecr_client, image_uri):
    """
    Uses ecr describe to generate all the tags of an image.

    :param ecr_client: boto3 Client for ECR
    :param image_uri: str Image URI
    :return: list, All the image tags
    """
    account_id = get_account_id_from_image_uri(image_uri)
    image_repo_name, image_tag = get_repository_and_tag_from_image_uri(image_uri)
    response = ecr_client.describe_images(
        registryId=account_id,
        repositoryName=image_repo_name,
        imageIds=[
            {
                'imageTag': image_tag
            },
        ]
    )
    return response['imageDetails'][0]['imageTags']

def get_framework_and_version_from_tag(image_uri):
    """
    Return the framework and version from the image tag.

    :param image_uri: ECR image URI
    :return: framework name, framework version
    """
    tested_framework = get_framework_from_image_uri(image_uri)
    allowed_frameworks = (
        "huggingface_tensorflow_trcomp",
        "huggingface_pytorch_trcomp",
        "huggingface_tensorflow",
        "huggingface_pytorch",
        "pytorch_trcomp"
        "tensorflow",
        "mxnet",
        "pytorch",
        "autogluon",
    )

    if not tested_framework:
        raise RuntimeError(
            f"Cannot find framework in image uri {image_uri} " f"from allowed frameworks {allowed_frameworks}"
        )

    tag_framework_version = re.search(r"(\d+(\.\d+){1,2})", image_uri).groups()[0]
    return tested_framework, tag_framework_version

def get_framework_from_image_uri(image_uri):
    return (
        "huggingface_tensorflow_trcomp"
        if "huggingface-tensorflow-trcomp" in image_uri
        else "huggingface_tensorflow"
        if "huggingface-tensorflow" in image_uri
        else "huggingface_pytorch_trcomp"
        if "huggingface-pytorch-trcomp" in image_uri
        else "pytorch_trcomp"
        if "pytorch-trcomp" in image_uri
        else "huggingface_pytorch"
        if "huggingface-pytorch" in image_uri
        else "mxnet"
        if "mxnet" in image_uri
        else "pytorch"
        if "pytorch" in image_uri
        else "tensorflow"
        if "tensorflow" in image_uri
        else "autogluon"
        if "autogluon" in image_uri
        else None
    )

def get_ecr_image_scan_results(ecr_client, image_uri, minimum_vulnerability="HIGH"):
    """
    Get list of vulnerabilities from ECR image scan results
    :param ecr_client:
    :param image_uri:
    :param minimum_vulnerability: str representing minimum vulnerability level to report in results
    :return: list<dict> Scan results
    """
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    scan_info = ecr_client.describe_image_scan_findings(repositoryName=repository, imageId={"imageTag": tag})
    
    scan_findings = [
        finding
        for finding in scan_info["imageScanFindings"]["findings"]
        if CVESeverity[finding["severity"]] >= CVESeverity[minimum_vulnerability]
    ]

    return scan_info

def reupload_image_to_test_ecr(source_image_uri, target_image_repo_name, target_region, pull_image=True):
    """
    Helper function to reupload an image owned by a another/same account to an ECR repo in this account to given region, so that
    this account can freely run tests without permission issues.

    :param source_image_uri: str Image URI for image to be tested
    :param target_image_repo_name: str Target image ECR repo name
    :param target_region: str Region where test is being run
    :param pull_image: bool, specifies if the source_image needs to be pulled before reuploading
    :return: str New image URI for re-uploaded image
    """
    ECR_PASSWORD_FILE_PATH = os.path.join("/tmp", f"{get_unique_name_from_tag(source_image_uri)}.txt")
    sts_client = boto3.client("sts", region_name=target_region)
    target_ecr_client = boto3.client("ecr", region_name=target_region)
    target_account_id = sts_client.get_caller_identity().get("Account")
    image_account_id = get_account_id_from_image_uri(source_image_uri)
    image_region = get_region_from_image_uri(source_image_uri)
    image_repo_uri, image_tag = source_image_uri.split(":")
    _, image_repo_name = image_repo_uri.split("/")
    if not ecr_repo_exists(target_ecr_client, target_image_repo_name):
        raise ECRRepoDoesNotExist(
            f"Repo named {target_image_repo_name} does not exist in {target_region} on the account {target_account_id}"
        )

    target_image_uri = (
        source_image_uri.replace(image_region, target_region)
        .replace(image_repo_name, target_image_repo_name)
        .replace(image_account_id, target_account_id)
    )

    client = boto3.client("ecr", region_name=image_region)
    username, password = get_ecr_login_boto3(client, image_account_id, image_region)

    save_credentials_to_file(ECR_PASSWORD_FILE_PATH, password)

    # using ctx.run throws error on codebuild "OSError: reading from stdin while output is captured".
    # Also it throws more errors related to awscli if in_stream=False flag is added to ctx.run which needs more deep dive
    if pull_image:
        subprocess.check_output(
            f"cat {ECR_PASSWORD_FILE_PATH} | docker login -u {username} --password-stdin https://{image_account_id}.dkr.ecr.{image_region}.amazonaws.com && docker pull {source_image_uri}",
            shell=True,
            executable="/bin/bash",
        )
    subprocess.check_output(f"docker tag {source_image_uri} {target_image_uri}", shell=True, executable="/bin/bash")
    delete_file(ECR_PASSWORD_FILE_PATH)
    username, password = get_ecr_login_boto3(target_ecr_client, target_account_id, target_region)
    save_credentials_to_file(ECR_PASSWORD_FILE_PATH, password)
    subprocess.check_output(
        f"cat {ECR_PASSWORD_FILE_PATH} | docker login -u {username} --password-stdin https://{target_account_id}.dkr.ecr.{target_region}.amazonaws.com && docker push {target_image_uri}",
        shell=True,
        executable="/bin/bash",
    )
    delete_file(ECR_PASSWORD_FILE_PATH)

    return target_image_uri

def get_ecr_image_enhanced_scan_status(ecr_client, image_uri):
    """
    Get status of an ECR Enhanced image scan.
    :param ecr_client: boto3 client for ECR
    :param image_uri: image URI for image to be checked
    :return: tuple<str, str> Scan Status, Status Description
    """
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    scan_info = ecr_client.describe_image_scan_findings(
        repositoryName=repository, imageId={"imageTag": tag}, maxResults=1
    )
    return scan_info["imageScanStatus"]["status"], scan_info["imageScanStatus"]["description"]

def get_all_ecr_image_scan_results(ecr_client, image_uri, scan_info_finding_key="enhancedFindings"):
    """
    Get list of All vulnerabilities from ECR image scan results using pagination
    :param ecr_client: boto3 ecr client
    :param image_uri: str, image uri
    :return: list<dict> Scan results
    """
    scan_info_findings = []
    registry_id = get_account_id_from_image_uri(image_uri)
    repository, tag = get_repository_and_tag_from_image_uri(image_uri)
    paginator = ecr_client.get_paginator('describe_image_scan_findings')
    response_iterator = paginator.paginate(
        registryId=registry_id,
        repositoryName=repository,
        imageId={
            'imageTag': tag
        },
        PaginationConfig={
            'PageSize': 50,
        }
    )

    for page in response_iterator:
        if scan_info_finding_key in page["imageScanFindings"]:
            scan_info_findings += page["imageScanFindings"][scan_info_finding_key]
    LOGGER.info(f"[TotalVulnsFound] For image_uri: {image_uri} {len(scan_info_findings)} vulnerabilities found in total.")
    return scan_info_findings

def get_all_ecr_enhanced_scan_findings(ecr_client, image_uri):
    """
    Get list of all vulnerabilities from ECR ENHANCED image scan results
    :param ecr_client:
    :param image_uri:
    :return: list<dict> Scan results
    """
    scan_info_findings = get_all_ecr_image_scan_results(
        ecr_client, image_uri, scan_info_finding_key="enhancedFindings"
    )
    return scan_info_findings

def wait_for_enhanced_scans_to_complete(ecr_client, image):
    """
    For Continuous Enhanced scans, the images will go through `SCAN_ON_PUSH` when they are uploaded for the 
    first time. During that time, their state will be shown as `PENDING`. From next time onwards, their status will show 
    itself as `ACTIVE`.

    :param ecr_client: boto3 Client for ECR
    :param image: str, Image URI for image being scanned
    """
    scan_status = None
    scan_status_description = ""
    start_time = time()
    while (time() - start_time) <= 30 * 60:
        try:
            scan_status, scan_status_description = get_ecr_image_enhanced_scan_status(ecr_client, image)
        except ecr_client.exceptions.ScanNotFoundException as e:
            LOGGER.info(e.response)
            LOGGER.info(
                "It takes sometime for the newly uploaded image to show its scan status, hence the error handling"
            )
        if scan_status == "ACTIVE":
            break
        sleep(4 * 60)
    if scan_status != "ACTIVE":
        raise TimeoutError(
            f"ECR Scan is still in {scan_status} state with description: {scan_status_description}. Exiting."
        )

class CVESeverity(IntEnum):
    UNDEFINED = 0
    INFORMATIONAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

def uniquify_list_of_complex_datatypes(list_of_complex_datatypes):
    assert all(type(element) == type(list_of_complex_datatypes[0]) for element in list_of_complex_datatypes), f"{list_of_complex_datatypes} has multiple types"
    if list_of_complex_datatypes:
        if isinstance(list_of_complex_datatypes[0], dict):
            return uniquify_list_of_dict(list_of_complex_datatypes)
        if dataclasses.is_dataclass(list_of_complex_datatypes[0]):
            type_of_dataclass = type(list_of_complex_datatypes[0])
            list_of_dict = json.loads(json.dumps(list_of_complex_datatypes, cls= EnhancedJSONEncoder))
            uniquified_list = uniquify_list_of_dict(list_of_dict=list_of_dict)
            return [type_of_dataclass(**uniquified_list_dict_element) for uniquified_list_dict_element in uniquified_list]
        raise "Not implemented"
    return list_of_complex_datatypes

def uniquify_list_of_dict(list_of_dict):
    """
    Takes list_of_dict as an input and returns a list of dict such that each dict is only present
    once in the returned list. Runs an operation that is similar to list(set(input_list)). However,
    for list_of_dict, it is not possible to run the operation directly.

    :param list_of_dict: List(dict)
    :return: List(dict)
    """
    list_of_string = [json.dumps(dict_element, sort_keys=True) for dict_element in list_of_dict]
    unique_list_of_string = list(set(list_of_string))
    unique_list_of_string.sort()
    list_of_dict_to_return = [json.loads(str_element) for str_element in unique_list_of_string]
    return list_of_dict_to_return

def get_allowlist_path_for_enhanced_scan_from_env_variable():
    return os.getenv("ALLOWLIST_PATH_ENHSCAN")

# {'registryId': '669063966089', 
# 'repositoryName': 'ecr-enhanced-scanning-dlc-repo',
#  'imageId': 
#     {'imageDigest': 'sha256:738a79afd3be5dcebdde2b190e984c59cf641b1513a155a50eac704077d524cd',
#      'imageTag': 'beta-pytorch-training-1.13.1-cpu-py39-ubuntu20.04-ec2-2023-02-02-20-17-55-pre-push-ENHSCAN'}, 
#      'imageScanStatus': {'status': 'ACTIVE', 
#      'description': 'Continuous scan is selected for image.
#     '},
# 'imageScanFindings':
#      {'imageScanCompletedAt': datetime.datetime(2023, 2, 2, 20, 23, 0, 994000, tzinfo=tzlocal()), 
#      'vulnerabilitySourceUpdatedAt': datetime.datetime(2023, 2, 2, 20, 23, 0, 994000, tzinfo=tzlocal()), 
#      'findingSeverityCounts': {'MEDIUM': 7, 'LOW': 11}
#      }, 
# 'ResponseMetadata': 
#     {'RequestId': 'aa0b392d-142c-4cbb-8b30-84c82ddbd9dc', 
#     'HTTPStatusCode': 200,
#      'HTTPHeaders': {'x-amzn-requestid': 'aa0b392d-142c-4cbb-8b30-84c82ddbd9dc', 'date': 'Thu, 02 Feb 2023 20:45:57 GMT', 'content-type': 'application/x-amz-json-1.1', 'content-length': '56360'}, 'RetryAttempts': 0}}

def get_dockerfile_path_for_image(image_uri):
    """
    For a given image_uri, find the path within the repository to its corresponding dockerfile

    :param image_uri: str Image URI
    :return: str Absolute path to dockerfile
    """
    github_repo_path = os.path.abspath(os.path.curdir).split("test", 1)[0]

    framework, framework_version = get_framework_and_version_from_tag(image_uri)

    if "trcomp" in framework:
        # Replace the trcomp string as it is extracted from ECR repo name
        framework = framework.replace("_trcomp", "")
        framework_path = framework.replace("_", os.path.sep)
    elif "huggingface" in framework:
        framework_path = framework.replace("_", os.path.sep)
    elif "habana" in image_uri:
        framework_path = os.path.join("habana", framework)
    else:
        framework_path = framework

    job_type = get_job_type_from_image(image_uri)

    short_framework_version = re.search(r"(\d+\.\d+)", image_uri).group(1)

    framework_version_path = os.path.join(github_repo_path, framework_path, job_type, "docker", short_framework_version)
    if not os.path.isdir(framework_version_path):
        long_framework_version = re.search(r"\d+(\.\d+){2}", image_uri).group()
        framework_version_path = os.path.join(
            github_repo_path, framework_path, job_type, "docker", long_framework_version
        )
    python_version = re.search(r"py\d+", image_uri).group()

    python_version_path = os.path.join(framework_version_path, python_version)
    if not os.path.isdir(python_version_path):
        python_version_path = os.path.join(framework_version_path, "py3")

    device_type = get_processor_from_image_uri(image_uri)
    cuda_version = get_cuda_version_from_tag(image_uri)
    synapseai_version = get_synapseai_version_from_tag(image_uri)
    neuron_sdk_version = get_neuron_sdk_version_from_tag(image_uri)

    dockerfile_name = get_expected_dockerfile_filename(device_type, image_uri)

    dockerfiles_list = [
        path
        for path in glob(os.path.join(python_version_path, "**", dockerfile_name), recursive=True)
        if "example" not in path
    ]
    dockerfiles_list = os.path.join(python_version_path, dockerfile_name)

    if device_type in ["gpu", "hpu", "neuron"]:
        if len(dockerfiles_list) > 1:
            if device_type == "gpu" and not cuda_version:
                raise LookupError(
                    f"dockerfiles_list has more than one result, and needs cuda_version to be in image_uri to "
                    f"uniquely identify the right dockerfile:\n"
                    f"{dockerfiles_list}"
                )
            if device_type == "hpu" and not synapseai_version:
                raise LookupError(
                    f"dockerfiles_list has more than one result, and needs synapseai_version to be in image_uri to "
                    f"uniquely identify the right dockerfile:\n"
                    f"{dockerfiles_list}"
                )
            if device_type == "neuron" and not neuron_sdk_version:
                raise LookupError(
                    f"dockerfiles_list has more than one result, and needs neuron_sdk_version to be in image_uri to "
                    f"uniquely identify the right dockerfile:\n"
                    f"{dockerfiles_list}"
                )
        if dockerfiles_list:
            if cuda_version:
                if cuda_version in dockerfiles_list:
                    return dockerfiles_list
            elif synapseai_version:
                if synapseai_version in dockerfiles_list:
                    return dockerfiles_list
            elif neuron_sdk_version:
                if neuron_sdk_version in dockerfiles_list:
                    return dockerfiles_list
        raise LookupError(f"Failed to find a dockerfile path for {cuda_version} in:\n{dockerfiles_list}")
        # for dockerfile_path in dockerfiles_list:
        #     if cuda_version:
        #         if cuda_version in dockerfile_path:
        #             return dockerfile_path
        #     elif synapseai_version:
        #         if synapseai_version in dockerfile_path:
        #             return dockerfile_path
        #     elif neuron_sdk_version:
        #         if neuron_sdk_version in dockerfile_path:
        #             return dockerfile_path
        # raise LookupError(f"Failed to find a dockerfile path for {cuda_version} in:\n{dockerfiles_list}")

    # assert len(dockerfiles_list) == 1, f"No unique dockerfile path in:\n{dockerfiles_list}\nfor image: {image_uri}"

    return dockerfiles_list

def get_job_type_from_image(image_uri):
    """
    Return the Job type from the image tag.

    :param image_uri: ECR image URI
    :return: Job Type
    """
    tested_job_type = None
    allowed_job_types = ("training", "inference")
    for job_type in allowed_job_types:
        if job_type in image_uri:
            tested_job_type = job_type
            break

    if not tested_job_type and "eia" in image_uri:
        tested_job_type = "inference"

    if not tested_job_type:
        raise RuntimeError(
            f"Cannot find Job Type in image uri {image_uri} " f"from allowed frameworks {allowed_job_types}"
        )

    return tested_job_type

def get_processor_from_image_uri(image_uri):
    """
    Return processor from the image URI

    Assumes image uri includes -<processor> in it's tag, where <processor> is one of cpu, gpu or eia.

    :param image_uri: ECR image URI
    :return: cpu, gpu, eia, neuron or hpu
    """
    allowed_processors = ["eia", "neuron", "cpu", "gpu", "hpu"]

    for processor in allowed_processors:
        match = re.search(rf"-({processor})", image_uri)
        if match:
            return match.group(1)
    raise RuntimeError("Cannot find processor")

def get_cuda_version_from_tag(image_uri):
    """
    Return the cuda version from the image tag as cuXXX
    :param image_uri: ECR image URI
    :return: cuda version as cuXXX
    """
    cuda_framework_version = None
    cuda_str = ["cu", "gpu"]
    image_region = get_region_from_image_uri(image_uri)
    ecr_client = boto3.Session(region_name=image_region).client('ecr')
    # all_image_tags = get_all_the_tags_of_an_image_from_ecr(ecr_client, image_uri)
    tag = image_uri.split(":")[1]


    if all(keyword in tag for keyword in cuda_str):
        cuda_framework_version = re.search(r"(cu\d+)-", tag).groups()[0]
        return cuda_framework_version

    if "gpu" in image_uri:
        raise CudaVersionTagNotFoundException()
    else:
        return None

def get_synapseai_version_from_tag(image_uri):
    """
    Return the synapseai version from the image tag.
    :param image_uri: ECR image URI
    :return: synapseai version
    """
    synapseai_version = None

    synapseai_str = ["synapseai", "hpu"]
    if all(keyword in image_uri for keyword in synapseai_str):
        synapseai_version = re.search(r"synapseai(\d+(\.\d+){2})", image_uri).groups()[0]

    return synapseai_version

def get_neuron_sdk_version_from_tag(image_uri):
    """
    Return the neuron sdk version from the image tag.
    :param image_uri: ECR image URI
    :return: neuron sdk version
    """
    neuron_sdk_version = None

    if "sdk" in image_uri:
        neuron_sdk_version = re.search(r"sdk([\d\.]+)", image_uri).group(1)

    return neuron_sdk_version

def get_expected_dockerfile_filename(device_type, image_uri):
    # if is_covered_by_ec2_sm_split(image_uri):
    #     print("here please 0")
    #     if "graviton" in image_uri:
    #         print("here please 1")
    #         return f"Dockerfile.graviton.{device_type}"
    #     elif is_ec2_image(image_uri):
    #         print("here please 5")
    #         return f"Dockerfile.ec2.{device_type}"
    #     elif is_ec2_sm_in_same_dockerfile(image_uri):
    #         print("here please 2")
    #         if "pytorch-trcomp-training" in image_uri:
    #             print("here please 3")
    #             return f"Dockerfile.trcomp.{device_type}"
    #         else:
    #             print("here please 4")
    #             return f"Dockerfile.{device_type}"
    #     else:
    #         print("here please 6")
    #         return f"Dockerfile.sagemaker.{device_type}"

    ## TODO: Keeping here for backward compatibility, should be removed in future when the
    ## functions is_covered_by_ec2_sm_split and is_ec2_sm_in_same_dockerfile are made exhaustive
    
    if is_ec2_image(image_uri):
        return f"Dockerfile.ec2.{device_type}"
    if is_sagemaker_image(image_uri):
        return f"Dockerfile.sagemaker.{device_type}"
    if is_trcomp_image(image_uri):
        return f"Dockerfile.trcomp.{device_type}"
    return f"Dockerfile.{device_type}"

def is_ec2_image(image_uri):
    return "-ec2" in image_uri


def is_sagemaker_image(image_uri):
    return "-sagemaker" in image_uri


def is_trcomp_image(image_uri):
    return "-trcomp" in image_uri

def is_covered_by_ec2_sm_split(image_uri):
    ec2_sm_split_images = {
        "pytorch": SpecifierSet(">=1.10.0"),
        "tensorflow": SpecifierSet(">=2.7.0"),
        "pytorch_trcomp": SpecifierSet(">=1.12.0"),
        "mxnet": SpecifierSet(">=1.9.0"),
    }
    framework, version = get_framework_and_version_from_tag(image_uri)
    return framework in ec2_sm_split_images and Version(version) in ec2_sm_split_images[framework]

def is_ec2_sm_in_same_dockerfile(image_uri):
    same_sm_ec2_dockerfile_record = {
        "pytorch": SpecifierSet(">=1.11.0"),
        "tensorflow": SpecifierSet(">=2.8.0"),
        "pytorch_trcomp": SpecifierSet(">=1.12.0"),
        "mxnet": SpecifierSet(">=1.9.0"),
    }
    framework, version = get_framework_and_version_from_tag(image_uri)
    return framework in same_sm_ec2_dockerfile_record and Version(version) in same_sm_ec2_dockerfile_record[framework]

def check_if_two_dictionaries_are_equal(dict1, dict2, ignore_keys=[]):
    """
    Compares if 2 dictionaries are equal or not. The ignore_keys argument is used to provide
    a list of keys that are ignored while comparing the dictionaries.

    :param dict1: dict
    :param dict2: dict
    :param ignore_keys: list[str], keys that are ignored while comparison
    """
    dict1_filtered = {k: v for k, v in dict1.items() if k not in ignore_keys}
    dict2_filtered = {k: v for k, v in dict2.items() if k not in ignore_keys}
    return dict1_filtered == dict2_filtered

class ECRScanFailedError(Exception):
    pass

class ECRRepoDoesNotExist(Exception):
    pass
