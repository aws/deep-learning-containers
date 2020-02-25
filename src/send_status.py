import os
import argparse

import utils

from github import GitHubHandler


def get_args():
    """
    Created a parser for explanation of code snippet usage
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--status",
                        choices=['0', '1', '2'],
                        help="Github status to set. 0 is fail, 1 is success, 2 is pending")
    parser.add_argument("--description",
                        default="I'm a PR build",
                        help="Description associated with status")

    return parser.parse_args()


def main():
    codebuild_statuses = {
        '0': 'failure',
        '1': 'success',
        '2': 'pending'
    }
    args = get_args()

    handler = GitHubHandler()
    project_name = utils.get_codebuild_project_name()
    trigger_job = os.getenv('TEST_TRIGGER')

    if not trigger_job:
        handler.set_status(state='error', context=project_name, description="Unknown test trigger")

    context = f"{project_name}_{trigger_job}"

    handler.set_status(state=codebuild_statuses[args.status], context=context, description=args.description)


if __name__ == "__main__":
    main()
