"""
Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"). You
may not use this file except in compliance with the License. A copy of
the License is located at

    http://aws.amazon.com/apache2.0/

or in the "license" file accompanying this file. This file is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.
"""
import os

import json
import requests
import boto3


class GitHubHandler:
    """
    Methods to handle interacting with GitHub
    """

    GITHUB_API_URL = "https://api.github.com"
    OAUTH_TOKEN = "/codebuild/github/oauth"

    def __init__(self, user='aws', repo='deep-learning-containers'):
        self.user = user
        self.repo = repo
        self.commit_hash = os.getenv('CODEBUILD_RESOLVED_SOURCE_VERSION')

    def get_auth_token(self):
        client = boto3.client("secretsmanager")
        resp = client.get_secret_value(SecretId=self.OAUTH_TOKEN)
        return resp['SecretString']


    def get_authorization_header(self):
        token = self.get_auth_token()
        return {"Authorization": "token {}".format(token)}

    def set_status(self, state, **kwargs):
        """

        Args:
            state: success, failure, pending, error
            **kwargs: common parameters - target_url, description, context

        Returns:
            requests object

        """
        url = f"{self.GITHUB_API_URL}/repos/{self.user}/{self.repo}/statuses/{self.commit_hash}"
        data = {"state": state}

        for key, value in kwargs.items():
            data[key] = value

        headers = self.get_authorization_header()

        return requests.post(url, headers=headers, data=json.dumps(data))

    def get_latest_sha(self, pull_request):
        """
        Get most recent sha from the PR

        Returns: <str> sha ID of commit
        """
        pr_status = self.get_pr_status(pull_request)
        return pr_status.json()["head"]["sha"]

    def get_pr_status(self, pull_request):
        """
        Get the whole status from a given PR

        Returns: full response object from PR
        """
        url = (
            f"{self.GITHUB_API_URL}/repos/{self.user}/{self.repo}/pulls/{pull_request}"
        )
        return requests.get(url)

    def get_pr_files_changed(self, pull_request):
        """
        Get the list of files changed in a PR

        Returns: List of filenames
        """

        print(f"PR: {pull_request}")

        url = (
            f"{self.GITHUB_API_URL}/repos/{self.user}/{self.repo}/pulls/{pull_request}/files"
        )


        response = requests.get(url, headers=self.get_authorization_header())
        files_changed = json.loads(response.text)
        print(url)
        print(files_changed)
        return [_file['filename'] for _file in files_changed]


