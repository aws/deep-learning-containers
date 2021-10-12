import os
import logging
import sys

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


class PytestCache:
    def __init__(self, s3_client):
        self.s3_client = s3_client

    def download_pytest_cache(self, current_dir, commit_id, framework, version, build_context, test_type):
        local_file_path = f"{current_dir}/.pytest_cache/v/cache"
        s3_file_path = self.make_s3_path(commit_id, framework, version, build_context, test_type)
        if os.path.exists(f"{local_file_path}/lastfailed"):
            os.remove(f"{local_file_path}/lastfailed")
        else:
            os.makedirs(local_file_path, exist_ok=True)

        LOGGER.info(f"Downloading previous executions cache: {s3_file_path}/lastfailed")
        try:
            self.s3_client.download_file('dlc-test-execution-results-669063966089',
                                             f"{s3_file_path}/lastfailed",
                                             f"{local_file_path}/lastfailed")
        except Exception as e:
            LOGGER.info(f"Cache file wasn't downloaded: {e}")

    def upload_pytest_cache(self, current_dir, commit_id, framework, version, build_context, test_type):
        local_file_path = f"{current_dir}/.pytest_cache/v/cache"
        s3_file_path = self.make_s3_path(commit_id, framework, version, build_context, test_type)
        if os.path.exists(f"{local_file_path}/lastfailed"):
            LOGGER.info(f"Uploading current execution result to {s3_file_path}/lastfailed")
            try:
                self.s3_client.upload_file(f"{local_file_path}/lastfailed",
                                               "dlc-test-execution-results-669063966089",
                                               f"{s3_file_path}/lastfailed")
                LOGGER.info(f"Cache file uploaded")
            except Exception as e:
                LOGGER.info(f"Cache file wasn't uploaded because of error: {e}")
        else:
            LOGGER.info(f"No cache file was created")

    def make_s3_path(self, commit_id, framework, version, build_context, specific_test_type):
        return f"{commit_id}/{framework}/{version}/{build_context}/{specific_test_type}"