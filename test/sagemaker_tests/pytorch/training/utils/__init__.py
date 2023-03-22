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
from __future__ import absolute_import
from enum import Enum
import os
import botocore

class NightlyFeatureLabel(Enum):
    AWS_FRAMEWORK_INSTALLED = "aws_framework_installed"
    AWS_SMDEBUG_INSTALLED = "aws_smdebug_installed"
    AWS_SMDDP_INSTALLED = "aws_smddp_installed"
    AWS_SMMP_INSTALLED = "aws_smmp_installed"
    AWS_S3_PLUGIN_INSTALLED = "aws_s3_plugin_installed"

    
def _botocore_resolver():
    """
    Get the DNS suffix for the given region.
    :return: endpoint object
    """
    loader = botocore.loaders.create_loader()
    return botocore.regions.EndpointResolver(loader.load_data('endpoints'))


def get_ecr_registry(account, region):
    """
    Get prefix of ECR image URI
    :param account: Account ID
    :param region: region where ECR repo exists
    :return: AWS ECR registry
    """
    endpoint_data = _botocore_resolver().construct_endpoint('ecr', region)
    return '{}.dkr.{}'.format(account, endpoint_data['hostname'])


def is_nightly_context():
    return os.getenv("BUILD_CONTEXT") == "NIGHTLY" or os.getenv("NIGHTLY_PR_TEST_MODE", "false").lower() == "true"
