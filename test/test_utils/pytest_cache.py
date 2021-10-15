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

    def download_pytest_cache_from_s3_to_local(self, current_dir, commit_id, framework, version, build_context,
                                               test_type):
        local_file_path = f"{current_dir}/.pytest_cache/v/cache"
        s3_file_path = self.__make_s3_path(commit_id, framework, version, build_context, test_type)
        if os.path.exists(f"{local_file_path}/lastfailed"):
            os.remove(f"{local_file_path}/lastfailed")
        else:
            os.makedirs(local_file_path, exist_ok=True)
        self.__download_cache_from_s3(f"{s3_file_path}/lastfailed", f"{local_file_path}/lastfailed")

    def download_pytest_cache_from_s3_to_ec2(self, ec2_connection, path, commit_id, framework, version, build_context,
                                             test_type):
        file_path = f"{path}/.pytest_cache/v/cache"
        s3_file_path = self.__make_s3_path(commit_id, framework, version, build_context, test_type)
        self.__delete_file_on_ec2(ec2_connection, f"{file_path}/lastfailed")

        self.__download_cache_from_s3(f"{s3_file_path}/lastfailed", "lastfailed")
        self.__upload_cache_to_ec2(ec2_connection, "lastfailed", file_path)

    def upload_pytest_cache_from_ec2_to_s3(self,
                                           ec2_connection,
                                           path,
                                           commit_id,
                                           framework,
                                           version,
                                           build_context,
                                           test_type):
        ec2_path = f"{path}/.pytest_cache/v/cache"
        s3_file_path = self.__make_s3_path(commit_id, framework, version, build_context, test_type)
        self.__download_cache_from_ec2(ec2_connection, f"{ec2_path}/lastfailed", "tmp")
        self.__merge_2_execution_caches_and_save("tmp", "lastfailed", "lastfailed")
        self.__upload_cache_to_s3("lastfailed", f"{s3_file_path}/lastfailed")

    def upload_pytest_cache_from_local_to_s3(self, current_dir, commit_id, framework, version, build_context,
                                             test_type):
        local_file_path = f"{current_dir}/.pytest_cache/v/cache"
        s3_file_path = self.__make_s3_path(commit_id, framework, version, build_context, test_type)
        self.__upload_cache_to_s3(F"{local_file_path}/lastfailed", f"{s3_file_path}/lastfailed")

    def __make_s3_path(self, commit_id, framework, version, build_context, test_type):
        return f"{commit_id}/{framework}/{version}/{build_context}/{test_type}"

    def __upload_cache_to_s3(self, local_file, s3_file):
        if os.path.exists(f"{local_file}"):
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

    def __merge_2_execution_caches_and_save(self, a, b, save_to):
        if self.__is_file_exist_and_not_empty(a):
            with open(a) as tmp1:
                json1 = json.load(tmp1)
        else:
            json1 = {}
        if self.__is_file_exist_and_not_empty(b):
            with open(b) as tmp2:
                json2 = json.load(tmp2)
        else:
            json2 = {}

        merged_json = {**json1, **json2}
        if len(merged_json) != 0:
            with open(save_to, "w") as f:
                f.write(json.dumps(merged_json))

    def __is_file_exist_and_not_empty(self, file_path):
        return os.path.exists(file_path) and os.stat(file_path).st_size != 0

    def __download_cache_from_s3(self, s3_file, local_file):
        LOGGER.info(f"Downloading previous executions cache: {s3_file}")
        try:
            self.s3_client.download_file('dlc-test-execution-results-669063966089',
                                         f"{s3_file}",
                                         f"{local_file}")
        except Exception as e:
            LOGGER.info(f"Cache file wasn't downloaded: {e}")

    def __upload_cache_to_ec2(self, ec2_connection, local_file, ec2_file):
        try:
            ec2_connection.put(local_file, f"{ec2_file}")
        except Exception as e:
            LOGGER.info(f"Cache file wasn't downloaded: {e}")

    def __download_cache_from_ec2(self, ec2_connection, ec2_file, local_file):
        LOGGER.info(f"Downloading executions cache from ec2 instance")
        try:
            ec2_connection.get(f"{ec2_file}", local_file)
        except Exception as e:
            LOGGER.info(f"Cache file wasn't downloaded: {e}")

    def delete_file_on_ec2(self, ec2_connection, ec2_file):
        ec2_connection.run(f"rm -f {ec2_file}")
