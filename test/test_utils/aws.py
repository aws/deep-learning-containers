# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""AWS Session Manager for all AWS boto3 API resources"""

import boto3
from test_utils.constants import DEFAULT_REGION


class AWSSessionManager:
    def __init__(self, region=DEFAULT_REGION, profile_name=None):
        if profile_name:
            self.session = boto3.Session(profile_name=profile_name, region_name=region)
        else:
            self.session = boto3.Session(region_name=region)

        # Client API
        self.cloudwatch = self.session.client("cloudwatch")
        self.codebuild = self.session.client("codebuild")
        self.codepipeline = self.session.client("codepipeline")
        self.ec2 = self.session.client("ec2")
        self.events = self.session.client("events")
        self.iam = self.session.client("iam")
        self.resource_groups = self.session.client("resource-groups")
        self.scheduler = self.session.client("scheduler")
        self.secretsmanager = self.session.client("secretsmanager")
        self.sts = self.session.client("sts")
        self.s3 = self.session.client("s3")
        self.sagemaker = self.session.client("sagemaker")

        # Resource API
        self.iam_resource = self.session.resource("iam")
        self.s3_resource = self.session.resource("s3")
