import os


# Reason string for skipping tests in PR context
SKIP_PR_REASON = "Skipping test in PR context to speed up iteration time. Test will be run in nightly/release pipeline."


def is_pr_context():
    return os.getenv("BUILD_CONTEXT") == "PR"
