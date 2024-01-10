#!/usr/bin/env bash

# Copyright 2018-2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

if [[ "$1" = "train" ]]; then
     CURRENT_HOST=$(jq .current_host  /opt/ml/input/config/resourceconfig.json)
     sed -ie "s/PLACEHOLDER_HOSTNAME/$CURRENT_HOST/g" changehostname.c
     gcc -o changehostname.o -c -fPIC -Wall changehostname.c
     gcc -o libchangehostname.so -shared -export-dynamic changehostname.o -ldl
     LD_PRELOAD=/libchangehostname.so train
else
     eval "$@"
fi
