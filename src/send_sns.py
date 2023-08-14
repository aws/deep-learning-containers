import os
import xmltodict
import json
import boto3

def get_pytest_output():
    """
    Get pytest output from file.
    """
    pytest_result_directory = os.path.join(os.getcwd(), "test")
    # get all xml files in test directory
    files = [os.path.join(pytest_result_directory, f) for f in os.listdir(pytest_result_directory) if f.endswith(".xml")]
    #print(files)
    # parse xml files and save it to dict
    pytest_output_dict = {}
    #TODO: check fir multiple test xml files
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


def get_failed_tests():
    """
    Parse pytest output to get test results.
    """
    pytest_results = get_pytest_output()
    failed_tests = {}
    for f in pytest_results:
        failed_tests[f] = {}
        for test in pytest_results[f]["testsuites"]["testsuite"]["testcase"]:
            if "failure" in test:
                team_name = test["properties"]["property"]["@value"]
                if team_name not in failed_tests[f]:
                    failed_tests[f][team_name] = []
                test_data = {}
                test_name, ecr_image, instance_name = get_test_details(test["@name"])
                test_data["test_name"] = test_name
                test_data["ecr_image"] = ecr_image
                test_data[test["properties"]["property"]["@name"]] = test["properties"]["property"]["@value"]
                test_data["instance_name"] = instance_name
                test_data["test_path"] = test["@classname"].replace(".", "/") + "/" + test_name
                test_data["fail_message"] = test["failure"]["@message"]
                test_data["fail_full_message"] = test["failure"]["#text"]
                failed_tests[f][team_name].append(test_data)
    return failed_tests

def generate_sns_message():
    # Send SNS message to a topic
    sns_client = boto3.Session(region_name="us-west-2").client('sns')
    sns_client.publish(
        TopicArn='DLC-test-results',
        Message=json.dumps(get_failed_tests()),
        MessageStructure='json',
        Subject='Test Results')

def main():
    output = get_failed_tests()
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    main()