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

import logging
import multiprocessing
import os
import re

from collections import namedtuple

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_CONTENT_TYPE = 'application/json'
DEFAULT_ACCEPT_HEADER = 'application/json'
CUSTOM_ATTRIBUTES_HEADER = 'X-Amzn-SageMaker-Custom-Attributes'

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_port, '
                     'custom_attributes, request_content_type, accept_header, content_length')


def parse_request(req, rest_port, grpc_port, default_model_name, model_name=None):
    tfs_attributes = parse_tfs_custom_attributes(req)
    tfs_uri = make_tfs_uri(rest_port, tfs_attributes, default_model_name, model_name)

    if not model_name:
        model_name = tfs_attributes.get('tfs-model-name')

    context = Context(model_name,
                      tfs_attributes.get('tfs-model-version'),
                      tfs_attributes.get('tfs-method'),
                      tfs_uri,
                      grpc_port,
                      req.get_header(CUSTOM_ATTRIBUTES_HEADER),
                      req.get_header('Content-Type') or DEFAULT_CONTENT_TYPE,
                      req.get_header('Accept') or DEFAULT_ACCEPT_HEADER,
                      req.content_length)

    data = req.stream
    return data, context


def make_tfs_uri(port, attributes, default_model_name, model_name=None):
    log.info("sagemaker tfs attributes: \n{}".format(attributes))

    tfs_model_name = model_name or attributes.get("tfs-model-name", default_model_name)
    tfs_model_version = attributes.get('tfs-model-version')
    tfs_method = attributes.get('tfs-method', 'predict')

    uri = 'http://localhost:{}/v1/models/{}'.format(port, tfs_model_name)
    if tfs_model_version:
        uri += '/versions/' + tfs_model_version
    uri += ':' + tfs_method
    return uri


def parse_tfs_custom_attributes(req):
    attributes = {}
    header = req.get_header(CUSTOM_ATTRIBUTES_HEADER)
    if header:
        matches = re.findall(r"(tfs-[a-z\-]+=[^,]+)", header)
        attributes = dict(attribute.split("=") for attribute in matches)
    return attributes


def create_tfs_config_individual_model(model_name, base_path):
    config = 'model_config_list: {\n'
    config += '  config: {\n'
    config += '    name: "{}",\n'.format(model_name)
    config += '    base_path: "{}",\n'.format(base_path)
    config += '    model_platform: "tensorflow"\n'
    config += '  }\n'
    config += '}\n'
    return config


def create_tfs_config(
        tfs_default_model_name,
        tfs_config_path,
):
    models = find_models()
    if not models:
        raise ValueError('no SavedModel bundles found!')

    if tfs_default_model_name == 'None':
        default_model = os.path.basename(models[0])
        if default_model:
            tfs_default_model_name = default_model
            log.info('using default model name: {}'.format(tfs_default_model_name))
        else:
            log.info('no default model detected')

    # config (may) include duplicate 'config' keys, so we can't just dump a dict
    config = 'model_config_list: {\n'
    for m in models:
        config += '  config: {\n'
        config += '    name: "{}",\n'.format(os.path.basename(m))
        config += '    base_path: "{}",\n'.format(m)
        config += '    model_platform: "tensorflow"\n'
        config += '  }\n'
    config += '}\n'

    log.info('tensorflow serving model config: \n%s\n', config)

    with open(tfs_config_path, 'w') as f:
        f.write(config)


def tfs_command(tfs_grpc_port,
                tfs_rest_port,
                tfs_config_path,
                tfs_enable_batching,
                tfs_batching_config_file):
    cmd = "tensorflow_model_server " \
          "--port={} " \
          "--rest_api_port={} " \
          "--model_config_file={} " \
          "--max_num_load_retries=0 {}" \
        .format(tfs_grpc_port, tfs_rest_port, tfs_config_path,
                get_tfs_batching_args(tfs_enable_batching, tfs_batching_config_file))
    return cmd


def find_models():
    base_path = '/opt/ml/model'
    models = []
    for f in _find_saved_model_files(base_path):
        parts = f.split('/')
        if len(parts) >= 6 and re.match(r'^\d+$', parts[-2]):
            model_path = '/'.join(parts[0:-2])
            if model_path not in models:
                models.append(model_path)
        return models


def _find_saved_model_files(path):
    for e in os.scandir(path):
        if e.is_dir():
            yield from _find_saved_model_files(os.path.join(path, e.name))
        else:
            if e.name == 'saved_model.pb':
                yield os.path.join(path, e.name)


def get_tfs_batching_args(enable_batching, tfs_batching_config):
    if enable_batching:
        return "--enable_batching=true " \
               "--batching_parameters_file={}".format(tfs_batching_config)
    else:
        return ""


def create_batching_config(batching_config_file):
    class _BatchingParameter:
        def __init__(self, key, env_var, value, defaulted_message):
            self.key = key
            self.env_var = env_var
            self.value = value
            self.defaulted_message = defaulted_message

    cpu_count = multiprocessing.cpu_count()
    batching_parameters = [
        _BatchingParameter('max_batch_size', 'SAGEMAKER_TFS_MAX_BATCH_SIZE', 8,
                           "max_batch_size defaulted to {}. Set {} to override default. "
                           "Tuning this parameter may yield better performance."),
        _BatchingParameter('batch_timeout_micros', 'SAGEMAKER_TFS_BATCH_TIMEOUT_MICROS', 1000,
                           "batch_timeout_micros defaulted to {}. Set {} to override "
                           "default. Tuning this parameter may yield better performance."),
        _BatchingParameter('num_batch_threads', 'SAGEMAKER_TFS_NUM_BATCH_THREADS',
                           cpu_count, "num_batch_threads defaulted to {},"
                                      "the number of CPUs. Set {} to override default."),
        _BatchingParameter('max_enqueued_batches', 'SAGEMAKER_TFS_MAX_ENQUEUED_BATCHES',
                           # Batch limits number of concurrent requests, which limits number
                           # of enqueued batches, so this can be set high for Batch
                           100000000 if 'SAGEMAKER_BATCH' in os.environ else cpu_count,
                           "max_enqueued_batches defaulted to {}. Set {} to override default. "
                           "Tuning this parameter may be necessary to tune out-of-memory "
                           "errors occur."),
    ]

    warning_message = ''
    for batching_parameter in batching_parameters:
        if batching_parameter.env_var in os.environ:
            batching_parameter.value = os.environ[batching_parameter.env_var]
        else:
            warning_message += batching_parameter.defaulted_message.format(
                batching_parameter.value, batching_parameter.env_var)
            warning_message += '\n'
    if warning_message:
        log.warning(warning_message)

    config = ''
    for batching_parameter in batching_parameters:
        config += '%s { value: %s }\n' % (batching_parameter.key, batching_parameter.value)

    log.info('batching config: \n%s\n', config)
    with open(batching_config_file, 'w') as f:
        f.write(config)
