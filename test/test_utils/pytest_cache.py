import json
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

    def download_pytest_cache_from_s3_to_ec2(self, ec2_connection, path, s3_file_path):
        file_path = f"{path}/.pytest_cache/v/cache"
        ec2_connection.run("mkdir -p /.pytest_cache && "
                           "mkdir -p /.pytest_cache/v && "
                           "mkdir -p /.pytest_cache/v/cache && "
                           "rm -f /.pytest_cache/v/cache/lastfailed")

        LOGGER.info(f"Downloading previous executions cache: {s3_file_path}/lastfailed")
        try:
            self.s3_client.download_file('dlc-test-execution-results-669063966089',
                                         f"{s3_file_path}/lastfailed",
                                         "lastfailed")
            ec2_connection.put("lastfailed", f"{file_path}")
        except Exception as e:
            LOGGER.info(f"Cache file wasn't downloaded: {e}")

    def upload_pytest_cache_from_ec2_to_s3(self,
                                           ec2_connection,
                                           path,
                                           commit_id,
                                           framework,
                                           version,
                                           build_context,
                                           test_type):
        ec2_path = f"{path}/.pytest_cache/v/cache"
        s3_file_path = self.make_s3_path(commit_id, framework, version, build_context, test_type)
        LOGGER.info(f"Downloading executions cache from ec2 instance")
        try:
            ec2_connection.get(f"{ec2_path}/lastfailed", "tmp")
        except Exception as e:
            LOGGER.info(f"Cache file wasn't downloaded: {e}")

        if os.path.exists(f"tmp"):
            with open("tmp") as tmp:
                tmp_results = json.load(tmp)
            if os.path.exists(f"lastfailed"):
                with open("lastfailed") as l:
                    lastfailed = json.load(l)
            else:
                lastfailed = {}
            lastfailed.update(tmp_results)
            with open("lastfailed", "w") as f:
                f.write(json.dumps(lastfailed))

        self.upload_to_s3("lastfailed", f"{s3_file_path}/lastfailed")

    def upload_pytest_cache(self, current_dir, commit_id, framework, version, build_context, test_type):
        local_file_path = f"{current_dir}/.pytest_cache/v/cache"
        s3_file_path = self.make_s3_path(commit_id, framework, version, build_context, test_type)
        if os.path.exists(f"{local_file_path}/lastfailed"):
            self.upload_to_s3(
                f"{local_file_path}/lastfailed",
                f"{s3_file_path}/lastfailed"
            )

    def append_lastailed_and_upload_pytest_cache_to_s3(self,
                                                       local_path,
                                                       commit_id,
                                                       framework,
                                                       version,
                                                       build_context,
                                                       test_type):
        local_path = f"{local_path}/.pytest_cache/v/cache"
        s3_file_path = self.make_s3_path(commit_id, framework, version, build_context, test_type)
        LOGGER.info(f"Appending executions cache")
        try:
            ec2_connection.get(f"{local_path}/lastfailed", "tmp")
        except Exception as e:
            LOGGER.info(f"Cache file wasn't downloaded: {e}")

        if os.path.exists(f"tmp"):
            with open("tmp") as tmp:
                tmp_results = json.load(tmp)
            if os.path.exists(f"lastfailed"):
                with open("lastfailed") as l:
                    lastfailed = json.load(l)
            else:
                lastfailed = {}
            lastfailed.update(tmp_results)
            with open("lastfailed", "w") as f:
                f.write(json.dumps(lastfailed))

        self.upload_to_s3("lastfailed", f"{s3_file_path}/lastfailed")

    def make_s3_path(self, commit_id, framework, version, build_context, test_type):
        return f"{commit_id}/{framework}/{version}/{build_context}/{test_type}"

    def upload_to_s3(self, local_file, s3_file):
        if os.path.exists(f"{local_file}/lastfailed"):
            LOGGER.info(f"Uploading current execution result to {s3_file}")
            try:
                self.s3_client.upload_file(local_file,
                                           "dlc-test-execution-results-669063966089",
                                           s3_file)
                LOGGER.info(f"Cache file uploaded")
            except Exception as e:
                LOGGER.info(f"Cache file wasn't uploaded because of error: {e}")
        else:
            LOGGER.info(f"No cache file was created")

    def merge_2_execution_caches(self, a, b, result):
        if os.path.exists(a):
            with open(a) as tmp1:
                json1 = json.load(tmp1)
        else:
            json1 = {}
        if os.path.exists(b):
            with open(b) as tmp2:
                json2 = json.load(tmp2)
        else:
            json2 = {}

        merged_json = {**json1, **json2}
        with open(result, "w") as f:
            f.write(json.dumps(merged_json))
