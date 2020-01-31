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

VERSION = '1.13.1'
REPO = 'sagemaker-tensorflow-scriptmode'
PY2_CPU_BINARY = 'https://s3-us-west-2.amazonaws.com/tensorflow-aws/1.13/AmazonLinux/cpu/latest-patch-latest-patch/tensorflow-1.13.1-cp27-cp27mu-linux_x86_64.whl' # noqa
PY3_CPU_BINARY = 'https://s3-us-west-2.amazonaws.com/tensorflow-aws/1.13/AmazonLinux/cpu/latest-patch-latest-patch/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl' # noqa
PY2_GPU_BINARY = 'https://s3-us-west-2.amazonaws.com/tensorflow-aws/1.13/AmazonLinux/gpu/latest-patch-latest-patch/tensorflow-1.13.1-cp27-cp27mu-linux_x86_64.whl' # noqa
PY3_GPU_BINARY = 'https://s3-us-west-2.amazonaws.com/tensorflow-aws/1.13/AmazonLinux/gpu/latest-patch-latest-patch/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl' # noqa
DEV_ACCOUNT = '142577830533'
REGION = 'us-west-2'


def _parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--account', type=str, default=DEV_ACCOUNT)
    parser.add_argument('--region', type=str, default=REGION)
    parser.add_argument('--version', type=str, default=VERSION)
    parser.add_argument('--py2-cpu-binary', type=str, default=PY2_CPU_BINARY)
    parser.add_argument('--py3-cpu-binary', type=str, default=PY3_CPU_BINARY)
    parser.add_argument('--py2-gpu-binary', type=str, default=PY2_GPU_BINARY)
    parser.add_argument('--py3-gpu-binary', type=str, default=PY3_GPU_BINARY)
    parser.add_argument('--repo', type=str, default=REPO)

    return parser.parse_args()


args = _parse_args()
binaries = {
    'py2-cpu': args.py2_cpu_binary,
    'py3-cpu': args.py3_cpu_binary,
    'py2-gpu': args.py2_gpu_binary,
    'py3-gpu': args.py3_gpu_binary
}
build_dir = os.path.join('docker', args.version)

# Run docker-login so we can pull the cached image
login_cmd = subprocess.check_output(
    'aws ecr get-login --no-include-email --registry-id {}'.format(args.account).split())
print('Executing docker login command: '.format(login_cmd))
subprocess.check_call(login_cmd.split())

for arch in ['cpu', 'gpu']:
    for py_version in ['2', '3']:

        binary_url = binaries['py{}-{}'.format(py_version, arch)]
        binary_file = os.path.basename(binary_url)
        cmd = 'wget -O {}/{} {}'.format(build_dir, binary_file, binary_url)
        print('Downloading binary file: {}'.format(cmd))
        subprocess.check_call(cmd.split())

        tag = '{}-{}-py{}'.format(args.version, arch, py_version)
        prev_image_uri = '{}.dkr.ecr.{}.amazonaws.com/{}:{}'.format(args.account, args.region, args.repo, tag)
        dockerfile = os.path.join(build_dir, 'Dockerfile.{}'.format(arch))

        tar_file_name = subprocess.check_output('ls {}/sagemaker_tensorflow_container*'.format(build_dir),
                                                shell=True).strip().decode('ascii')
        print('framework_support_installable is {}'.format(os.path.basename(tar_file_name)))

        build_cmd = 'docker build -f {} --cache-from {} --build-arg framework_support_installable={} ' \
                    '--build-arg py_version={} --build-arg framework_installable={} ' \
                    '-t {}:{} {}'.format(dockerfile, prev_image_uri, os.path.basename(tar_file_name), py_version,
                                         binary_file, args.repo, tag, build_dir)
        print('Building docker image: {}'.format(build_cmd))
        subprocess.check_call(build_cmd.split())

        print('Deleting binary file {}'.format(binary_file))
        subprocess.check_call('rm {}'.format(os.path.join(build_dir, binary_file)).split())
