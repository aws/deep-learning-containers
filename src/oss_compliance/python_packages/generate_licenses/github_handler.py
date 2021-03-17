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
import base64
import requests
import boto3


class GitHubHandler:
    """
    Methods to handle interacting with GitHub
    """

    GITHUB_API_URL = "https://api.github.com"
   
    def __init__(self, user="aws", repo="deep-learning-containers"):
        self.user = user
        self.repo = repo
        self.__client = boto3.client("secretsmanager")

    def get_file_contents(self, filename, tag):
        """
        Get the list of files changed in a PR
        Returns: List of filenames
        """
        tags = [ tag, "v{}".format(tag)]
        # resp = self.__client.get_secret_value(SecretId="/codebuild/github/oauth")
        # token = resp["SecretString"]
        headers = {"Authorization": f"token "}
        content = ""
        for tag in tags:
            url = f"{self.GITHUB_API_URL}/repos/{self.user}/{self.repo}/contents/{filename}?ref={tag}"
            response = requests.get(url, headers=headers)
            # print(response.status_code)
            response_json = json.loads(response.text)
            if "patsy" in url:
                print(response_json)
            if response.status_code == 200:
                try:
                    if isinstance(response_json, dict):
                        content = base64.b64decode(response_json['content']).decode()
                    else:
                        for json_entry in response_json:
                            new_response = requests.get(json_entry['url'], headers=headers)
                            new_response_json = json.loads(new_response.text)
                            content += base64.b64decode(new_response_json['content']).decode()
                            content += "\n\n"
                except Exception as e:
                    print(e)
        
        if content == "":
            url = f"{self.GITHUB_API_URL}/repos/{self.user}/{self.repo}/contents/{filename}?ref=master"
            response = requests.get(url, headers=headers)
            # print(response.status_code)
            response_json = json.loads(response.text)
            if "patsy" in url:
                print(response_json)
            if response.status_code == 200:
                try:
                    if isinstance(response_json, dict):
                        content = base64.b64decode(response_json['content']).decode()
                    else:
                        for json_entry in response_json:
                            new_response = requests.get(json_entry['url'], headers=headers)
                            new_response_json = json.loads(new_response.text)
                            content += base64.b64decode(new_response_json['content']).decode()
                            content += "\n\n"
                except Exception as e:
                    print(e)

                
        return content


# gith = GitHubHandler("nucleic", "kiwi")
# print(gith.get_file_contents("LICENSE", "1.3.1"))


# urls = [
#     "https://github.com/chaimleib/intervaltree",
#     "https://github.com/chaimleib/intervaltree/"
# ]

# for url in urls:
#     url_split =  ' '.join(url.split('/')).split()
#     print(url_split[-1])
#     print(url_split[-2])
