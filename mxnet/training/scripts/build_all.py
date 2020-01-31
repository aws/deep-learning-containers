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
import os
import subprocess

DEFAULT_REGION = 'us-west-2'


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--account')
    parser.add_argument('--version')
    parser.add_argument('--repo')
    parser.add_argument('--region', default=DEFAULT_REGION)

    return parser.parse_args()


def _build_image(build_dir, arch, prev_image_uri, py_version):
    if py_version == '2':
        ls_tar_cmd = 'ls {}'.format(os.path.join('dist', 'sagemaker_mxnet_container*.tar.gz'))
        tar_file = subprocess.check_output(ls_tar_cmd, shell=True).strip().decode('ascii')
        print('framework_support_installable: {}'.format(os.path.basename(tar_file)))

        dockerfile = os.path.join(build_dir, 'Dockerfile.{}'.format(arch))

        build_cmd = [
            'docker', 'build',
            '-f', dockerfile,
            '--cache-from', prev_image_uri,
            '--build-arg', 'framework_support_installable={}'.format(tar_file),
            '-t', dest,
            '.',
        ]

        print('Building docker image: {}'.format(' '.join(build_cmd)))
        subprocess.check_call(build_cmd)
    else:
        dockerfile = 'Dockerfile.{}'.format(arch)

        build_cmd = [
            'docker', 'build',
            '-f', dockerfile,
            '--cache-from', prev_image_uri,
            '-t', dest,
            '.',
        ]

        prev_dir = os.getcwd()
        os.chdir(build_dir)

        print('Building docker image: {}'.format(' '.join(build_cmd)))
        subprocess.check_call(build_cmd)

        os.chdir(prev_dir)


args = _parse_args()

root_build_dir = os.path.join('docker', args.version)

# Run docker-login so we can pull the cached image
login_cmd = subprocess.check_output(
    'aws ecr get-login --no-include-email --registry-id {}'.format(args.account).split())
print('Executing docker login command: '.format(login_cmd))
subprocess.check_call(login_cmd.split())

for arch in ['cpu', 'gpu']:
    for py_version in ['2', '3']:
        tag = '{}-{}-py{}'.format(args.version, arch, py_version)
        dest = '{}:{}'.format(args.repo, tag)
        prev_image_uri = '{}.dkr.ecr.{}.amazonaws.com/{}'.format(args.account, args.region, dest)

        build_dir = os.path.join(root_build_dir, 'py{}'.format(py_version))
        _build_image(build_dir, arch, prev_image_uri, py_version)
