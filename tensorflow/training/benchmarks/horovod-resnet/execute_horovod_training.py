#!/usr/bin/env python
# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import itertools
import json
import os
import shutil
import subprocess

import click
import pandas as pd
from sagemaker import Session
from sagemaker.tensorflow import TensorFlow

dir_path = os.path.dirname(os.path.realpath(__file__))
benchmark_results_dir = os.path.join('s3://', Session().default_bucket(), 'hvd-benchmarking')


@click.group()
def cli():
    pass


def generate_report():
    results_dir = os.path.join(dir_path, 'results')

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    subprocess.call(['aws', 's3', 'cp', '--recursive', benchmark_results_dir, results_dir])

    jobs = {}

    for job_name in os.listdir(results_dir):
        jobs[job_name] = {}

        _, instance_type, instance_count, device, py_version, _, _, _, _, _, _, _ = job_name.split('-')

        current_dir = os.path.join(results_dir, job_name)

        model_dir = os.path.join(current_dir, 'output', 'model.tar.gz')
        subprocess.call(['tar', '-xvzf', model_dir], cwd=current_dir)

        jobs[job_name]['instance_type'] = instance_type
        jobs[job_name]['instance_count'] = instance_count
        jobs[job_name]['device'] = device
        jobs[job_name]['py_version'] = py_version

        benchmark_log = os.path.join(current_dir, 'benchmark_run.log')

        if os.path.exists(benchmark_log):
            with open(benchmark_log) as f:
                data = json.load(f)


                jobs[job_name]['dataset'] = data['dataset']['name']
                jobs[job_name]['num_cores'] = data['machine_config']['cpu_info']['num_cores']
                jobs[job_name]['cpu_info'] = data['machine_config']['cpu_info']['cpu_info']
                jobs[job_name]['mhz_per_cpu'] = data['machine_config']['cpu_info']['mhz_per_cpu']
                jobs[job_name]['gpu_count'] = data['machine_config']['gpu_info']['count']
                jobs[job_name]['gpu_model'] = data['machine_config']['gpu_info']['model']

                def find_value(parameter):
                    other_key = [k for k in parameter if k != 'name'][0]
                    return parameter[other_key]

                for parameter in data['run_parameters']:
                    jobs[job_name][parameter['name']] = find_value(parameter)

                jobs[job_name]['model_name'] = data['model_name']
                jobs[job_name]['run_date'] = data['run_date']
                jobs[job_name]['tensorflow_version'] = data['tensorflow_version']['version']
                jobs[job_name]['tensorflow_version_git_hash'] = data['tensorflow_version']['git_hash']

    return pd.DataFrame(jobs)


@cli.command('train')
@click.option('--framework-version', required=True, type=click.Choice(['1.11', '1.12']))
@click.option('--device', required=True, type=click.Choice(['cpu', 'gpu']))
@click.option('--py-versions', multiple=True, type=str)
@click.option('--training-input-mode', default='File', type=click.Choice(['File', 'Pipe']))
@click.option('--networking-isolation/--no-networking-isolation', default=False)
@click.option('--wait/--no-wait', default=False)
@click.option('--security-groups', multiple=True, type=str)
@click.option('--subnets', multiple=True, type=str)
@click.option('--role', default='SageMakerRole', type=str)
@click.option('--instance-counts', multiple=True, type=int)
@click.option('--instance-types', multiple=True, type=str)
@click.argument('script_args', nargs=-1, type=str)
def train(framework_version,
          device,
          py_versions,
          training_input_mode,
          networking_isolation,
          wait,
          security_groups,
          subnets,
          role,
          instance_counts,
          instance_types,
          script_args):
    iterator = itertools.product(instance_types, py_versions, instance_counts)
    for instance_type, py_version, instance_count in iterator:
        base_name = job_name(instance_type, instance_count, device, py_version)

        mpi_options = '-x HOROVOD_HIERARCHICAL_ALLREDUCE=1 -x HOROVOD_FUSION_THRESHOLD=16777216 -x TF_CPP_MIN_LOG_LEVEL=0 -x HOROVOD_TIMELINE --output-filename /opt/ml/model/hlog'
        estimator = TensorFlow(
            entry_point=os.path.join(dir_path, 'train.sh'),
            role=role,
            dependencies=[os.path.join(dir_path, 'train_imagenet_resnet_hvd.py')],
            base_job_name=base_name,
            train_instance_count=instance_count,
            train_instance_type=instance_type,
            framework_version=framework_version,
            py_version=py_version,
            script_mode=True,
            hyperparameters={
                'sagemaker_mpi_enabled': True,
                'sagemaker_mpi_num_of_processes_per_host': 8,
                'sagemaker_mpi_custom_mpi_options': mpi_options
            },
            output_path=benchmark_results_dir,
            security_group_ids=security_groups,
            subnets=subnets
        )

        estimator.fit(wait=wait)

        if wait:
            artifacts_path = os.path.join(dir_path, 'results',
                                          estimator.latest_training_job.job_name)
            model_path = os.path.join(artifacts_path, 'model.tar.gz')
            os.makedirs(artifacts_path)
            subprocess.call(['aws', 's3', 'cp', estimator.model_data, model_path])
            subprocess.call(['tar', '-xvzf', model_path], cwd=artifacts_path)

            print('Model downloaded at %s' % model_path)


def job_name(instance_type,
             instance_count,
             device,
             python_version):
    instance_typename = instance_type.replace('.', '').replace('ml', '')

    return 'hvd-%s-%s-%s-%s' % (
        instance_typename, instance_count, device, python_version)

if __name__ == '__main__':
    cli()
