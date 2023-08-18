import os
import xmltodict
import json
import boto3
from send_status import get_target_url

from codebuild_environment import (
    get_codebuild_project_name,
    get_codebuild_project_id,
    get_codepipeline_url,
)


def get_pytest_output():
    """
    Get pytest output from file.
    """
    pytest_result_directory = os.path.join(os.getcwd(), "test")
    # get all xml files in test directory
    files = [
        os.path.join(pytest_result_directory, f)
        for f in os.listdir(pytest_result_directory)
        if f.endswith(".xml")
    ]
    # parse xml files and save it to list
    pytest_output_dict = {}
    if files:
        for f in files:
            with open(f, "r") as xml_file:
                pytest_output_dict[f] = xmltodict.parse(xml_file.read())

    return pytest_output_dict


def get_test_details(name):
    test_name = name.split("[")[0]
    repo_instance_name = name.split("[")[1].replace("]", "")

    instance_name = repo_instance_name.split("-")[-1]
    ecr_image = repo_instance_name.replace(f"-{instance_name}", "")

    return test_name, ecr_image, instance_name


def get_dlc_images(build_context):
    if build_context == "PR":
        return os.getenv("DLC_IMAGES")
    if build_context == "MAINLINE":
        test_env_file = os.path.join(
            os.getenv("CODEBUILD_SRC_DIR_DLC_IMAGES_JSON"), "test_type_images.json"
        )
        with open(test_env_file) as test_env:
            test_images = json.load(test_env)
        for dlc_test_type, images in test_images.items():
            if dlc_test_type == "sanity":
                return " ".join(images)
        raise RuntimeError(f"Cannot find any images for in {test_images}")


def get_platform_execution_details():
    platform_details = {}
    build_context = os.getenv("BUILD_CONTEXT")
    codebuild_name = get_codebuild_project_name()

    platform_details["platform_info"] = {}

    platform_details["platform_info"]["build_context"] = build_context
    platform_details["platform_info"]["dlc_images"] = get_dlc_images(build_context)
    platform_details["platform_info"]["test_type"] = os.getenv("TEST_TYPE")
    platform_details["platform_info"]["codebuild_name"] = codebuild_name
    platform_details["platform_info"]["codebuild_id"] = get_codebuild_project_id()
    platform_details["platform_info"]["codebuild_url"] = get_target_url(codebuild_name)

    if build_context == "PR":
        pr_execution_details = get_pr_execution_details()
        platform_details["platform_info"]["PR"] = pr_execution_details
    elif build_context == "MAINLINE":
        mainline_execution_details = get_mainline_execution_details()
        platform_details["platform_info"]["MAINLINE"] = mainline_execution_details

    return platform_details


def get_pr_execution_details():
    pr_execution_details = {}
    pr_number = os.getenv("PR_NUMBER")
    github_url = os.getenv("CODEBUILD_SOURCE_REPO_URL")
    pr_execution_details["pr_number"] = pr_number
    pr_execution_details["commit_id"] = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    pr_execution_details["github_url"] = github_url
    pr_execution_details["pr_url"] = f"{github_url}/pull/{pr_number}"
    return pr_execution_details


def get_mainline_execution_details():
    mainline_execution_details = {}
    codepipeline_name = os.getenv("CODEBUILD_INITIATOR").split("/")[-1]
    mainline_execution_details["codepipeline_name"] = codepipeline_name
    mainline_execution_details["codepipeline_execution_id"] = os.getenv("CODEPIPELINE_EXECUTION_ID")
    mainline_execution_details["code_pipeline_url"] = get_codepipeline_url(codepipeline_name)


def parse_pytest_data():
    """
    Parse pytest output to get test results.
    """
    pytest_raw_data = get_pytest_output()
    pytest_parsed_output = []

    for f in pytest_raw_data:
        pytest_file_data = {}
        pytest_file_data["file_name"] = f
        pytest_file_data["failed_tests"] = {}
        for test in pytest_raw_data[f]["testsuites"]["testsuite"]["testcase"]:
            if "failure" in test:
                team_name = test["properties"]["property"]["@value"]
                if team_name not in pytest_file_data["failed_tests"]:
                    pytest_file_data["failed_tests"][team_name] = []
                test_data = {}
                test_name, ecr_image, instance_name = get_test_details(test["@name"])
                test_data["test_name"] = test_name
                test_data["ecr_image"] = ecr_image
                test_data[test["properties"]["property"]["@name"]] = test["properties"]["property"][
                    "@value"
                ]
                test_data["instance_name"] = instance_name
                test_data["test_path"] = test["@classname"].replace(".", "/") + "/" + test_name
                test_data["fail_message"] = test["failure"]["@message"]
                # test_data["fail_full_message"] = test["failure"]["#text"]
                pytest_file_data["failed_tests"][team_name].append(test_data)
        pytest_parsed_output.append(pytest_file_data)
    return pytest_parsed_output


def generate_sns_message_body():
    """
    Generate SNS message body.
    """
    sns_message_body = get_platform_execution_details()
    sns_message_body["pytest_output"] = parse_pytest_data()
    return sns_message_body


def generate_sns_message():
    # Send SNS message to a topic
    sns_client = boto3.Session(region_name="us-west-2").client("sns")
    print("publishing")
    sns_client.publish(
        TopicArn="arn:aws:sns:us-west-2:332057208146:AsimovAutoCutTicket-Service-332057208146-AsimovAutoCutTicketTopic0CE1A907-0iEfow67RJrS",
        Message=json.dumps(generate_sns_message_body()),
        Subject="Test Results",
    )


def main():
    output = generate_sns_message_body()
    with open("sns-body.json", "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
