import os
import boto3
from invoke import run


def pull_image_locally_with_all_its_tags_attached(image_uri):
    from test import test_utils
    run(f"docker pull {image_uri}")
    image_region = test_utils.get_region_from_image_uri(image_uri=image_uri)
    ecr_client = boto3.client("ecr", region_name=image_region)
    image_repo = image_uri.split(":")[0]
    tag_list = test_utils.get_all_the_tags_of_an_image_from_ecr(ecr_client=ecr_client, image_uri=image_uri)
    for tag in tag_list:
        run(f"docker tag {image_uri} {image_repo}:{tag}")
    return tag_list


def get_push_time_of_image_from_ecr(image_uri):
    from test import test_utils
    image_region = test_utils.get_region_from_image_uri(image_uri=image_uri)
    ecr_client = boto3.client("ecr", region_name=image_region)
    return test_utils.get_image_push_time_from_ecr(ecr_client=ecr_client, image_uri=image_uri)


def get_benchmark_tag_attached_to_the_latest_image_in_beta(autopatch_image_tag_list):
    benchmark_tag_list = [tag for tag in autopatch_image_tag_list if tag.endswith("-benchmark-tested")]
    assert len(benchmark_tag_list) == 1, f"{benchmark_tag_list} has multiple or no benchmark tested image tag"
    return benchmark_tag_list[0].replace("-autopatch","")


def get_benchmark_tested_image_uri_for_beta_image(autopatch_image_uri, benchmark_tag_in_beta):
    ap_image_repo = autopatch_image_uri.split(":")[0]
    beta_image_repo = ap_image_repo.replace("/autopatch-", "/beta-")
    return f"{beta_image_repo}:{benchmark_tag_in_beta}"


def main():
    from test import test_utils

    dlc_images = test_utils.get_dlc_images()
    image_list = dlc_images.split(" ")
    for autopatch_image in image_list:
        print(autopatch_image)
        autopatch_image_tag_list = pull_image_locally_with_all_its_tags_attached(image_uri=autopatch_image)
        benchmark_tag_in_beta = get_benchmark_tag_attached_to_the_latest_image_in_beta(autopatch_image_tag_list=autopatch_image_tag_list)
        beta_latest_benchmark_image_uri = get_benchmark_tested_image_uri_for_beta_image(autopatch_image_uri=autopatch_image, benchmark_tag_in_beta=benchmark_tag_in_beta)
        beta_image_push_time = get_push_time_of_image_from_ecr(image_uri=beta_latest_benchmark_image_uri)
        print(benchmark_tag_in_beta)
        print(beta_image_push_time)
        print(type(beta_image_push_time))


if __name__ == "__main__":
    main()
