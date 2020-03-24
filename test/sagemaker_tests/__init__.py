import os


def is_pr_context():
    return os.getenv("BUILD_CONTEXT") == "PR"
