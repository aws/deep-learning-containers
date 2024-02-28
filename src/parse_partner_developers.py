import argparse
import logging
import os
import sys

from config import get_dlc_developer_config_path


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))
LOGGER.addHandler(logging.StreamHandler(sys.stderr))


def main():
    toml_path = get_dlc_developer_config_path()
    pr_number = os.getenv("PR_NUMBER")
    try:
        handler = GitHubHandler()

        if pr_number:
            pr_status = handler.get_pr_status(pr_number)
            pr_body = pr_status.json()["body"]
            found_stack = ["```toml", "```"]
            with open(toml_path, "w"):
                for line in pr_body.split("\n"):
                    if not found_stack:
                        break
                    elif line == found_stack[0]:
                        char = found_stack.pop(0)
                        LOGGER.info(char)
                    elif len(found_stack) == 1:
                        toml_path.write(line)
                        LOGGER.info(line)

    except Exception as err:
        LOGGER.info(
            f"UNABLE TO PARSE TOML FROM PR BODY. DEFAULTING TO TOML IN REPO. FULL ERROR: {err}"
        )


if __name__ == "__main__":
    main()
