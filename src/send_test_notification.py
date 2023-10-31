import os
import xmltodict
import json
import config
from send_status import get_target_url
from codebuild_environment import get_cloned_folder_path
from dlc.ticket_notification_handler import TicketNotificationHandler

from codebuild_environment import (
    get_codebuild_project_name,
    get_codebuild_project_id,
    get_codepipeline_url,
    get_cloudwatch_url,
)


def get_pytest_output():
    """
    Get pytest output from file.
    """
    pytest_result_directory = os.path.join(os.getcwd(), "test")
    # get all xml files in test directory
    files = [
        os.path.join(pytest_result_directory, file)
        for file in os.listdir(pytest_result_directory)
        if file.endswith(".xml")
    ]
    # parse xml files and save it to list
    pytest_output_dict = {}
    if files:
        for file in files:
            with open(file, "r") as xml_file:
                pytest_output_dict[file] = xmltodict.parse(xml_file.read())

    return pytest_output_dict


def get_test_details(name):
    test_type = os.getenv("TEST_TYPE")
    test_name = name.split("[")[0]
    if "ec2" in test_type.lower():
        repo_instance_name = name.split("[")[1].replace("]", "")

        instance_name = repo_instance_name.split("-")[-1]
        ecr_image = repo_instance_name.replace(f"-{instance_name}", "")

        return test_name, ecr_image, instance_name
    else:
        return test_name, None, None


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


def get_platform_execution_details(build_context):
    platform_details = {}
    platform_details["platform_info"] = {}
    codebuild_name = get_codebuild_project_name()

    if build_context == "PR":
        pr_execution_details = get_pr_execution_details()
        platform_details["platform_info"]["PR"] = pr_execution_details
    elif build_context == "MAINLINE":
        mainline_execution_details = get_mainline_execution_details()
        platform_details["platform_info"]["MAINLINE"] = mainline_execution_details
    else:
        raise RuntimeError(f"Invalid build context {build_context}")

    platform_details["platform_info"]["build_context"] = build_context
    platform_details["platform_info"]["dlc_images"] = get_dlc_images(build_context)
    platform_details["platform_info"]["test_type"] = os.getenv("TEST_TYPE")
    platform_details["platform_info"]["codebuild_name"] = codebuild_name
    platform_details["platform_info"]["codebuild_id"] = get_codebuild_project_id()
    platform_details["platform_info"]["codebuild_url"] = get_target_url(codebuild_name)
    platform_details["platform_info"]["cloudwatch_logs_url"] = get_cloudwatch_url(codebuild_name)

    return platform_details


def get_pr_execution_details():
    pr_execution_details = {}
    pr_number = os.getenv("PR_NUMBER")
    github_url = os.getenv("CODEBUILD_SOURCE_REPO_URL")
    pr_execution_details["pr_number"] = pr_number
    pr_execution_details["commit_id"] = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    pr_execution_details["github_url"] = github_url
    pr_execution_details["pr_url"] = f"{github_url}/pull/{pr_number}"

    if config.is_notify_test_failures_enabled:
        pr_execution_details["notification_severity"] = config.get_notification_severity()

    return pr_execution_details


def get_mainline_execution_details():
    mainline_execution_details = {}
    codepipeline_name = os.getenv("CODEBUILD_INITIATOR").split("/")[-1]
    mainline_execution_details["codepipeline_name"] = codepipeline_name
    mainline_execution_details["codepipeline_execution_id"] = os.getenv("CODEPIPELINE_EXECUTION_ID")
    mainline_execution_details["code_pipeline_url"] = get_codepipeline_url(codepipeline_name)
    return mainline_execution_details


def get_allowlisted_test_exception():    
    test_exception_allowlist_file = os.path.join(
        os.sep, get_cloned_folder_path(), "data", "test-exception-allowlist.json"
    )

    with open(test_exception_allowlist_file) as f:
        allowlisted_exception = json.load(f)
    
    return allowlisted_exception.get("infrastructure_exceptions", [])


def check_for_infrastructure_exceptions(fail_message):
    allowlisted_exceptions = get_allowlisted_test_exception()
    for exception in allowlisted_exceptions:
        if exception in fail_message:
            return True
    return False

def parse_pytest_data():
    """
    Parse pytest output to get test results.
    """
    pytest_raw_data = get_pytest_output()
    pytest_parsed_output = []

    for file in pytest_raw_data:
        pytest_file_data = {}
        pytest_file_data["file_name"] = file
        pytest_file_data["failed_tests"] = {}
        for test in pytest_raw_data[file]["testsuites"]["testsuite"]["testcase"]:
            if "failure" in test:
                team_name = test["properties"]["property"]["@value"]
                if team_name not in pytest_file_data["failed_tests"]:
                    pytest_file_data["failed_tests"][team_name] = []
                test_data = {}
                test_name, ecr_image, instance_name = get_test_details(test["@name"])
                test_data["test_name"] = test_name
                if ecr_image is not None:
                    test_data["ecr_image"] = ecr_image
                if instance_name is not None:
                    test_data["instance_name"] = instance_name
                test_data[test["properties"]["property"]["@name"]] = test["properties"]["property"][
                    "@value"
                ]
                test_data["test_path"] = test["@classname"].replace(".", "/") + "/" + test_name
                test_data["fail_message"] = test["failure"]["@message"]

                fail_full_message = test["failure"]["#text"]
                if check_for_infrastructure_exceptions(fail_full_message):
                    print("Infrastructure failure found in the test. Skipping test details")
                else:
                    pytest_file_data["failed_tests"][team_name].append(test_data)
       
        failed_test_for_file = pytest_file_data["failed_tests"].copy()
        for team_name in failed_test_for_file:
            if not failed_test_for_file[team_name]:
                del pytest_file_data["failed_tests"][team_name]
        
        if pytest_file_data["failed_tests"]:
            pytest_parsed_output.append(pytest_file_data)
    return pytest_parsed_output


def generate_test_execution_data(build_context):
    """
    Generate test execution data.
    """
    test_execution_data = get_platform_execution_details(build_context)
    test_execution_data["pytest_output"] = parse_pytest_data()
    return test_execution_data


def main():
    build_context = os.getenv("BUILD_CONTEXT")
    if build_context == "MAINLINE" or (
        build_context == "PR" and config.is_notify_test_failures_enabled()
    ):
        print("Sending test notification...")
        test_execution_data = generate_test_execution_data(build_context)
        handler = TicketNotificationHandler()
        handler.publish_notification(test_execution_data)
    else:
        print("Test notification is disabled.")


if __name__ == "__main__":
    main()
