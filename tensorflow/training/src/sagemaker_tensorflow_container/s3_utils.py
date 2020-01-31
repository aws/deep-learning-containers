# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os

import boto3
from six.moves.urllib.parse import urlparse


def configure(model_dir, job_region):

    os.environ['S3_REGION'] = _s3_region(job_region, model_dir)

    # setting log level to WARNING
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['S3_USE_HTTPS'] = '1'


def _s3_region(job_region, model_dir):
    if model_dir and model_dir.startswith('s3://'):
        s3 = boto3.client('s3', region_name=job_region)

        # We get the AWS region of the checkpoint bucket, which may be different from
        # the region this container is currently running in.
        parsed_url = urlparse(model_dir)
        bucket_name = parsed_url.netloc

        bucket_location = s3.get_bucket_location(Bucket=bucket_name)['LocationConstraint']

        return bucket_location or job_region
    else:
        return job_region
