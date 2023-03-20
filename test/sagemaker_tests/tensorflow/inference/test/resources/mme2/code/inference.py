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

import json


def handler(data, context):
    """A default inference request handler that directly send post request to TFS rest port with
    un-processed data and return un-processed response

    :param data: input data
    :param context: context instance that contains tfs_rest_uri
    :return: inference response from TFS model server
    """
    data = data.read().decode("utf-8")
    if not isinstance(data, str):
        data = json.loads(data)
    response = requests.post(context.rest_uri, data=data)
    return response.content, context.accept_header
