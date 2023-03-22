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

import os
import subprocess
import sys

CYAN_COLOR = '\033[36m'
END_COLOR = '\033[0m'


def build_base_image(framework_name, framework_version, py_version,
                     processor, base_image_tag, cwd='.'):
    base_image_uri = get_base_image_uri(framework_name, base_image_tag)

    dockerfile_location = os.path.join('docker', framework_version, 'base',
                                       'Dockerfile.{}'.format(processor))

    subprocess.check_call(['docker', 'build', '-t', base_image_uri,
                           '-f', dockerfile_location, '--build-arg',
                           'py_version={}'.format(py_version[-1]), cwd], cwd=cwd)
    print('created image {}'.format(base_image_uri))
    return base_image_uri


def get_base_image_uri(framework_name, base_image_tag):
    return '{}-base:{}'.format(framework_name, base_image_tag)


def get_image_uri(framework_name, tag):
    return '{}:{}'.format(framework_name, tag)


def _check_call(cmd, *popenargs, **kwargs):
    if isinstance(cmd, str):
        cmd = cmd.split(" ")
    _print_cmd(cmd)
    subprocess.check_call(cmd, *popenargs, **kwargs)


def _print_cmd(cmd):
    print('executing docker command: {}{}{}'.format(CYAN_COLOR, ' '.join(cmd), END_COLOR))
    sys.stdout.flush()
