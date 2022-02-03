import json
import os
import logging
import sys

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


class PytestCache:
    """
    A handler for pytest cache
    Contains methods for uploading/dowloading pytest cache file to/from ec2 instances and s3 buckets
    """

    def __init__(self, s3_client, account_id):
        self.s3_client = s3_client
        self.bucket_name = f"dlc-test-execution-results-{account_id}"

    def download_pytest_cache_from_s3_to_local(
        self, current_dir, commit_id, framework, version, build_context, test_type, custom_cache_directory=""
    ):
        """
        Download pytest cache file from directory in s3 to local box

        :param current_dir: directory where the script is executed. .pytest_cache directory will be created in this
        local directory.
                Following parameters are required to create a path to cache file in s3:
        :param commit_id
        :param framework
        :param version
        :param build_context
        :param test_type
        :param custom_cache_directory - the prefix used to create custom pytest cache directories.
        """

        if custom_cache_directory:
            current_dir = os.path.join(current_dir, custom_cache_directory)
        local_file_dir = os.path.join(current_dir, ".pytest_cache", "v", "cache")
        local_file_path = os.path.join(local_file_dir, "lastfailed")
        s3_file_dir = self.__make_s3_path(commit_id, framework, version, build_context, test_type)
        s3_file_path = os.path.join(s3_file_dir, "lastfailed")

        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        else:
            os.makedirs(local_file_dir, exist_ok=True)
        self.__download_cache_from_s3(s3_file_path, local_file_path)

    def download_pytest_cache_from_s3_to_ec2(
        self, ec2_connection, path, commit_id, framework, version, build_context, test_type
    ):
        """
        Copy pytest cache file from directory in s3 to ec2 instance. .pytest_cache directory will be created in 
        :param path ec2 directory.

                Following parameters are required to create a path to cache file in s3:
        :param path: directory on ec2 instance
        :param commit_id
        :param framework
        :param version
        :param build_context
        :param test_type
        """
        local_file_dir = os.path.join(path, ".pytest_cache", "v", "cache")
        local_file_path = os.path.join(local_file_dir, "lastfailed")
        s3_file_dir = self.__make_s3_path(commit_id, framework, version, build_context, test_type)
        s3_file_path = os.path.join(s3_file_dir, "lastfailed")
        self.__delete_file_on_ec2(ec2_connection, local_file_path)

        self.__download_cache_from_s3(s3_file_path, "lastfailed")
        self.__upload_cache_to_ec2(ec2_connection, "lastfailed", local_file_dir)

    def upload_pytest_cache_from_ec2_to_s3(
        self, ec2_connection, path, commit_id, framework, version, build_context, test_type
    ):
        """
        Copy pytest cache file from ec2 instance to directory in s3. .pytest_cache directory will be copied from 
        :param path ec2 directory to s3 directory generated from parameters.

                Following parameters are required to create a path to cache file in s3:
        :param path: directory on ec2 instance
        :param commit_id
        :param framework
        :param version
        :param build_context
        :param test_type
        """
        ec2_dir = os.path.join(path, ".pytest_cache", "v", "cache")
        ec2_file_path = os.path.join(ec2_dir, "lastfailed")
        s3_file_dir = self.__make_s3_path(commit_id, framework, version, build_context, test_type)
        s3_file_path = os.path.join(s3_file_dir, "lastfailed")

        # Since we run tests in parallel files from latests executions will overwrite existing file.
        # So put the latest file into tmp, add it to local lastfailed and upload to s3.
        # At the end of current execution there will be full file in s3
        tmp_file_name = "tmp"
        self.__download_cache_from_ec2(ec2_connection, ec2_file_path, tmp_file_name)
        self.__merge_2_execution_caches_and_save(tmp_file_name, "lastfailed", "lastfailed")
        self.__upload_cache_to_s3("lastfailed", s3_file_path)

    def upload_pytest_cache_from_local_to_s3(
        self, current_dir, commit_id, framework, version, build_context, test_type
    ):
        """
        Copy pytest cache file from local box to directory in s3. .pytest_cache directory will be copied from 
        :param current_dir ec2 directory to s3 directory generated from parameters.

                Following parameters are required to create a path to cache file in s3:
        :param current_dir: directory on ec2 instance
        :param commit_id
        :param framework
        :param version
        :param build_context
        :param test_type
        """
        local_file_dir = os.path.join(current_dir, ".pytest_cache", "v", "cache")
        local_file_path = os.path.join(local_file_dir, "lastfailed")
        s3_file_dir = self.__make_s3_path(commit_id, framework, version, build_context, test_type)
        s3_file_path = os.path.join(s3_file_dir, "lastfailed")
        self.__upload_cache_to_s3(local_file_path, s3_file_path)

    def convert_cache_json_and_upload_to_s3(self, cache_json, commit_id, framework, version, build_context, test_type):
        """
        Copy pytest cache from json and send to directory in s3. 
        :param cache_json - json object with pytest cache
                Following parameters are required to create a path to cache file in s3:
        :param commit_id
        :param framework
        :param version
        :param build_context
        :param test_type
        """
        if not cache_json:
            LOGGER.info("No cache was generated. Skip uploading.")
            return
        s3_file_dir = self.__make_s3_path(commit_id, framework, version, build_context, test_type)
        s3_file_path = os.path.join(s3_file_dir, "lastfailed")
        tmp_file_for_cache_json = "tmp_file_for_cache_json"
        with open(tmp_file_for_cache_json, "w") as f:
            json.dump(dict(cache_json), f)
        self.__upload_cache_to_s3(tmp_file_for_cache_json, s3_file_path)

    def convert_pytest_cache_file_to_json(self, current_dir, custom_cache_directory=""):

        if custom_cache_directory:
            current_dir = os.path.join(current_dir, custom_cache_directory)
        local_file_dir = os.path.join(current_dir, ".pytest_cache", "v", "cache")
        local_file_path = os.path.join(local_file_dir, "lastfailed")
        return self.get_json_from_file(local_file_path)

    def __make_s3_path(self, commit_id, framework, version, build_context, test_type):
        return os.path.join(commit_id, framework, version, build_context, test_type)

    def __upload_cache_to_s3(self, local_file, s3_file):
        if os.path.exists(f"{local_file}"):
            LOGGER.info(f"Uploading current execution result to {s3_file}")
            try:
                self.s3_client.upload_file(local_file, self.bucket_name, s3_file)
                LOGGER.info(f"Cache file uploaded")
            except Exception as e:
                LOGGER.info(f"Cache file wasn't uploaded because of error: {e}")
        else:
            LOGGER.info(f"No cache file was created")

    def __merge_2_execution_caches_and_save(self, cache_file_1, cache_file_2, save_to):
        """
        Merges 2 JSON objects into one and safe on disk 

        :param cache_file_1
        :param cache_file_2
        :param save_to: filename where to save result JSON
        """
        json1 = self.get_json_from_file(cache_file_1)
        json2 = self.get_json_from_file(cache_file_2)

        merged_json = {**json1, **json2}
        if len(merged_json) != 0:
            with open(save_to, "w") as f:
                f.write(json.dumps(merged_json))

    def __is_file_exist_and_not_empty(self, file_path):
        return os.path.exists(file_path) and os.stat(file_path).st_size != 0

    def __download_cache_from_s3(self, s3_file, local_file):
        LOGGER.info(f"Downloading previous executions cache: {s3_file}")
        try:
            self.s3_client.download_file(self.bucket_name, f"{s3_file}", f"{local_file}")
        except Exception as e:
            LOGGER.info(f"Cache file wasn't downloaded: {e}")

    def __upload_cache_to_ec2(self, ec2_connection, local_file, ec2_file):
        try:
            ec2_connection.put(local_file, f"{ec2_file}")
        except Exception as e:
            LOGGER.info(f"Cache file wasn't uploaded: {e}")

    def __download_cache_from_ec2(self, ec2_connection, ec2_file, local_file):
        LOGGER.info(f"Downloading executions cache from ec2 instance")
        try:
            ec2_connection.get(f"{ec2_file}", local_file)
        except Exception as e:
            LOGGER.info(f"Cache file wasn't downloaded: {e}")

    def __delete_file_on_ec2(self, ec2_connection, ec2_file):
        ec2_connection.run(f"rm -f {ec2_file}")

    def get_json_from_file(self, file):
        if self.__is_file_exist_and_not_empty(file):
            with open(file) as tmp1:
                json_obj = json.load(tmp1)
        else:
            json_obj = {}
        return json_obj
