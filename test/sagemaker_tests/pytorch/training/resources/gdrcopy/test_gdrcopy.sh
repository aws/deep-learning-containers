#!/usr/bin/env bash

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

GDRCOPY_VERSION="$(awk '$1=="#define"&&$2=="GDR_API_MAJOR_VERSION" {printf "%s.", $3} $1=="#define"&&$2=="GDR_API_MINOR_VERSION" {printf "%s\n", $3}' /usr/local/include/gdrapi.h)"

function version { echo "$@" | awk -F. '{ printf("%d%03d%03d%03d\n", $1,$2,$3,$4); }'; }

if [ $(version $GDRCOPY_VERSION) -ge $(version "2.4") ]; then
  # Test GDRCopy version 2.4 and above
  gdrcopy_sanity > tmp_out
  if [ $? != 0 ]; then
      echo "GDRCopy Sanity check failed!"
      exit 1
  fi

  cat tmp_out | grep 'Failed: 0' &> /dev/null
  if [ $? != 0 ]; then
      echo "GDRCopy Sanity check passed but failed at grep output check!"
      echo "Please examine the gdrcopy_sanity output to ensure the tests are passing properly"
      exit 1
  fi
else
  # Test GDRCopy version below 2.4
  sanity | grep 'Failures: 0, Errors: 0' &> /dev/null
  if [ $? != 0 ]; then
      echo "GDRCopy Sanity check failed!"
      exit 1
  fi
fi
echo "GDRCopy Sanity check succeed!"
