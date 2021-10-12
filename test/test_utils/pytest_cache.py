import os
import logging
import sys

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


class PytestCache:
    def __init__(self, s3_client):
        self.s3_client = s3_client

    def download_pytest_cache(self, current_dir, commit_id, framework, version):
        if os.path.exists(f"{current_dir}/.pytest_cache/v/cache/lastfailed"):
            os.remove(f"{current_dir}/.pytest_cache/v/cache/lastfailed")
        else:
            os.makedirs(f"{current_dir}/.pytest_cache/v/cache", exist_ok=True)

        LOGGER.info(f"Downloading previous executions cache: {commit_id}/{framework}/{version}/lastfailed")
        try:
            self.s3_client.download_file('dlc-test-execution-results-669063966089',
                                             f"{commit_id}/{framework}/{version}/lastfailed",
                                             f"{os.curdir}/.pytest_cache/v/cache/lastfailed")
        except Exception as e:
            LOGGER.info(f"Cache file wasn't downloaded: {e}")

    def upload_pytest_cache(self, current_dir, commit_id, framework, version):
        if os.path.exists(f"{current_dir}/.pytest_cache/v/cache/lastfailed"):
            LOGGER.info(f"Uploading current execution result for commit: {commit_id}")
            try:
                self.s3_client.upload_file(f"{os.curdir}/.pytest_cache/v/cache/lastfailed",
                                               "dlc-test-execution-results-669063966089",
                                               f"{commit_id}/{framework}/{version}/lastfailed")
                LOGGER.info(f"Cache file uploaded")
            except Exception as e:
                LOGGER.info(f"Cache file wasn't uploaded because of error: {e}")
        else:
            LOGGER.info(f"No cache file was created")


