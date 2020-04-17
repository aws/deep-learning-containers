# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import tarfile


def make_tarfile(script, model, output_path, filename="model.tar.gz"):
    output_filename = os.path.join(output_path, filename)
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(script, arcname=os.path.basename(script))
        tar.add(model, arcname=os.path.basename(model))
    return output_filename
