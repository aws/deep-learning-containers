# Copyright 2018-2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import sys
import argparse
import subprocess
import re


def check_cmd(cmd):
    cmd_dir = subprocess.check_output(["which", cmd], text=True)
    if "command not found" in cmd_dir:
        print(cmd_dir, file=sys.stderr)
        return 1

    return 0


def check_sanity():
    if check_cmd("sanity"):
        return 1

    try:
        test_log = subprocess.check_output(["sanity"], text=True)
        num_checks = int(re.compile(r'[\s\S]*Checks: (\d+)').match(test_log).groups()[0])
        num_failures = int(re.compile(r'[\s\S]*Failures: (\d+)').match(test_log).groups()[0])
        num_errors = int(re.compile(r'[\s\S]*Errors: (\d+)').match(test_log).groups()[0])
    except:
        print("Failed gdrcopy sanity test.", file=sys.stderr)
        return 1

    if num_failures != 0 or num_errors != 0:
        print(test_log, file=sys.stderr)
        print("Failed gdrcopy sanity test.", file=sys.stderr)
        return 1
    else:
        print("Passed gdrcopy sanity test.")
        return 0


def check_copybw():
    if check_cmd("copybw"):
        return 1
    
    try:
        test_log_0 = subprocess.check_output(["copybw"], text=True)  # cuMemAlloc by default
        test_log_1 = subprocess.check_output(["copybw", "-a", "cuMemCreate"], text=True)
    except:
        print("Failed gdrcopy copybw test.", file=sys.stderr)
        return 1
    
    print("Passed gdrcopy copybw test.")
    return 0


def check_copylat():
    if check_cmd("copylat"):
        return 1
    
    try:
        test_log_0 = subprocess.check_output(["copylat"], text=True)  # cuMemAlloc by default
        test_log_1 = subprocess.check_output(["copylat", "-a", "cuMemCreate"], text=True)
    except:
        print("Failed gdrcopy copylat test.", file=sys.stderr)
        return 1
    
    print("Passed gdrcopy copylat test.")
    return 0


def main():
    ret = 0
    if "sanity" in args.types:
        ret |= check_sanity()
    if "copybw" in args.types:
        ret |= check_copybw()
    if "copylat" in args.types:
        ret |= check_copylat()
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", type=str, nargs="+", default=[])

    args = parser.parse_args()
    sys.exit(main())
