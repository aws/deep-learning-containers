import os
import argparse

import utils

from dlc.github_handler import GitHubHandler


def get_args():
    """
    Manage arguments to this script when called directly
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        choices=["0", "1", "2"],
        help="Github status to set. 0 is fail, 1 is success, 2 is pending",
    )
    return parser.parse_args()


def get_target_url(project):
    """
    Set the link for "Details" on PR builds

    :param project: CodeBuild project name associated with the running build
    :return: Link for the "Details" link associated with a GitHub status check
    """
    region = os.getenv("AWS_REGION")
    logpath = os.getenv("CODEBUILD_LOG_PATH")
    return f"https://{region}.console.aws.amazon.com/codesuite/codebuild/projects/{project}/build/{project}%3A{logpath}" \
           f"/log?region={region}"


def set_build_description(state, project):
    """
    Set the build description, based on the state, project name, and job that triggered the project.

    :param state: <str> choices are "success", "failure", "error" or "pending"
    :param project: Project name associated with the running CodeBuild job
    :return: <str> Description to be posted to the PR build
    """
    if state == "success":
        return f"{project} succeeded."
    elif state == "failure" or state == "error":
        return f"{project} is in state {state.upper()}! Check details to debug."
    elif state == "pending":
        return f"{project} is pending..."
    else:
        return f"Unknown state: {state}"


def post_status(state):
    """
    Post the status with a constructed context to the PR.

    :param state: <str> choices are "success", "failure", "error" or "pending"
    """
    # Executor does not send logs to GitHub when using the Scheduler
    if os.getenv("EXECUTOR_MODE", "False").lower() == "true":
        return

    project_name = utils.get_codebuild_project_name()
    target_url = get_target_url(project_name)

    test_context = os.getenv("TEST_TYPE")

    if test_context:
        trigger_job = os.getenv("TEST_TRIGGER", "UNKNOWN-TEST-TRIGGER")
        context = f"{trigger_job}_{project_name}"
    else:
        context = f"{project_name}"
    
    description = set_build_description(state, project_name)

    # Example: "https://github.com/aws/deep-learning-containers.git"
    repo_url = os.getenv("CODEBUILD_SOURCE_REPO_URL")
    _, user, repo_name = repo_url.rstrip(".git").rsplit("/", 2)

    handler = GitHubHandler(user, repo_name)
    handler.set_status(
        state=state,
        context=context,
        description=description,
        target_url=target_url
    )


def main():
    codebuild_statuses = {"0": "failure", "1": "success", "2": "pending"}
    args = get_args()

    state = codebuild_statuses[args.status]

    # Send status for given state
    if os.getenv("BUILD_CONTEXT") == "PR" and not os.getenv("CODEBUILD_WEBHOOK_TRIGGER", "").startswith("pr/"):
        post_status(state)


if __name__ == "__main__":
    main()
