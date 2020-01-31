# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import argparse
import subprocess

DEV_ACCOUNT = '142577830533'
VERSION = '1.13.1'
REGION = 'us-west-2'
REPO = 'sagemaker-tensorflow-scriptmode'


def _parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--account', type=str, default=DEV_ACCOUNT)
    parser.add_argument('--version', type=str, default=VERSION)
    parser.add_argument('--repo', type=str, default=REPO)
    parser.add_argument('--region', type=str, default=REGION)

    return parser.parse_args()


args = _parse_args()

for arch in ['cpu', 'gpu']:
    for py_version in ['2', '3']:
        source = '{}:{}-{}-py{}'.format(args.repo, args.version, arch, py_version)
        dest = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(args.account, args.region, source)
        tag_cmd = 'docker tag {} {}'.format(source, dest)
        print('Tagging image: {}'.format(tag_cmd))
        subprocess.check_call(tag_cmd.split())
        login_cmd = subprocess.check_output(
            'aws ecr get-login --no-include-email --registry-id {} --region {}'
            .format(args.account, args.region).split())
        print('Executing docker login command: {}'.format(login_cmd))
        subprocess.check_call(login_cmd.split())
        push_cmd = 'docker push {}'.format(dest)
        print('Pushing image: {}'.format(push_cmd))
        subprocess.check_call(push_cmd.split())
