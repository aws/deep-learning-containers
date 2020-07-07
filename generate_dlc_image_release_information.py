import argparse
import logging
import json
import os
import sys
import shutil
import tarfile

import boto3

from botocore.exceptions import ClientError
from release.dlc_release_information import DLCReleaseInformation

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.setLevel(logging.INFO)


def write_to_file(file_name, content):
    """
    Uses json package to dump content to a file. The content may or may-not be json-structured.
    :param file_name:
    :param content:
    :return:
    """
    with open(file_name, "w") as fp:
        fp.write(content)
        LOGGER.info(f"Content written to file: {file_name}")


def upload_to_S3(local_file_path, bucket_name, bucket_key):
    """
    Upload a local file to s3 with specified bucket name and key
    :param local_file_path: string
    :param bucket_name: string
    :param bucket_key: string
    :return:
    """
    _s3 = boto3.client('s3')
    try:
        _s3.upload_file(local_file_path, bucket_name, bucket_key)
    except ClientError as e:
        LOGGER.error("Error: Cannot upload file to s3 bucket.")
        LOGGER.error("Exception: {}".format(e))
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Specify bucket name to upload release information for DLC")
    parser.add_argument("--artifact-bucket", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    dlc_artifact_bucket = args.artifact_bucket

    dlc_tag = os.getenv("ASIMOV_TAG_WITH_VERSION")
    dlc_account_id = os.getenv("TARGET_ACCOUNT_ID_CLASSIC")
    dlc_repository = os.getenv("TARGET_ECR_REPOSITORY")
    dlc_region = os.getenv("REGION")

    if not dlc_tag:
        raise ValueError(
            "Environment variable ASIMOV_TAG_WITH_VERSION not set. This environment variable is expected to be set by the promoter stage.")

    dlc_release_information = DLCReleaseInformation(dlc_account_id, dlc_region, dlc_repository, dlc_tag)

    release_info = {
        "bom_pip_packages": dlc_release_information.bom_pip_packages,
        "bom_apt_packages": dlc_release_information.bom_apt_packages,
        "bom_pipdeptree": dlc_release_information.bom_pipdeptree,
        "imp_pip_packages": dlc_release_information.imp_pip_packages,
        "imp_apt_packages": dlc_release_information.imp_apt_packages,
        "image_digest": dlc_release_information.image_digest,
        "image_tags": dlc_release_information.image_tags
    }

    dlc_folder = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")

    directory = os.path.join(os.sep, os.getcwd(), dlc_folder)
    if directory != "" and not os.path.isdir(directory):
        LOGGER.info(f"Creating folder: {directory}")
        os.mkdir(directory)

    # Zip the BOM for pip and apt packages. Store the tar in S3, and save its s3URI in release_info as reference.
    bom_pip_packages_local_path = os.path.join(os.sep, directory, "bom_pip_packages.md")
    bom_apt_packages_local_path = os.path.join(os.sep, directory, "bom_apt_packages.md")
    bom_pipdeptree_local_path = os.path.join(os.sep, directory, "bom_pipdeptree.md")

    write_to_file(bom_pip_packages_local_path, release_info["bom_pip_packages"])
    write_to_file(bom_apt_packages_local_path, release_info["bom_apt_packages"])
    write_to_file(bom_pipdeptree_local_path, release_info["bom_pipdeptree"])

    # Save the zip as <dlc_tag>_BOM.zip
    tarfile_path = f"{dlc_tag}_BOM.tar.gz"
    with tarfile.open(tarfile_path, "w:gz") as dlc_bom_tar:
        dlc_bom_tar.add(bom_pip_packages_local_path, arcname=os.path.basename(bom_pip_packages_local_path))
        dlc_bom_tar.add(bom_apt_packages_local_path, arcname=os.path.basename(bom_apt_packages_local_path))
        dlc_bom_tar.add(bom_pipdeptree_local_path, arcname=os.path.basename(bom_pipdeptree_local_path))

    LOGGER.info(f"Created BOM zip: {tarfile_path}")

    upload_to_S3(tarfile_path, dlc_artifact_bucket, f"{dlc_folder}/{dlc_tag}/{tarfile_path}")

    s3URI = f"s3://{dlc_artifact_bucket}/{dlc_folder}/{dlc_tag}/"

    release_info["bom_tar_s3_uri"] = f"{s3URI}{tarfile_path}"

    dlc_release_info_json = os.path.join(os.sep, directory, "dlc_release_info.json")
    write_to_file(dlc_release_info_json, json.dumps(release_info))

    upload_to_S3(dlc_release_info_json, dlc_artifact_bucket, f"{dlc_folder}/{dlc_tag}/dlc_release_info.json")

    # Cleanup
    os.remove(tarfile_path)
    shutil.rmtree(directory)

    LOGGER.info(f"Release Information collected for image: {dlc_release_information.image}")
    LOGGER.info(f"Release information and BOM uploaded to: s3://{dlc_artifact_bucket}/{dlc_folder}/{dlc_tag}")
