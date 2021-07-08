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
from __future__ import absolute_import

import os

import pytest
import sagemaker.huggingface
from sagemaker.huggingface import HuggingFace
import botocore
from datasets.filesystems import S3FileSystem

from ...utils.local_mode_utils import assert_files_exist
from ...integration import ROLE, distrilbert_script


@pytest.mark.model("hf_bert")
@pytest.mark.integration("hf_local")
@pytest.mark.skip_cpu
@pytest.mark.skip_py2_containers
def test_distilbert_base(docker_image, processor, instance_type, sagemaker_local_session, py_version):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # tokenizer used in preprocessing
    tokenizer_name = 'distilbert-base-uncased'

    # dataset used
    dataset_name = 'imdb'

    # s3 key prefix for the data
    s3_prefix = 'samples/datasets/imdb'
    # load dataset
    dataset = load_dataset(dataset_name)

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    # load dataset
    train_dataset, test_dataset = load_dataset('imdb', split=['train', 'test'])
    test_dataset = test_dataset.shuffle().select(range(100))  # smaller the size for test dataset to 10k

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))

    # set format for pytorch
    train_dataset =  train_dataset.rename_column("label", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset = test_dataset.rename_column("label", "labels")
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # hyperparameters, which are passed into the training job
    hyperparameters = {'max_steps': 5,
                       'train_batch_size': 4,
                       'model_name': 'distilbert-base-uncased'
                       }

    s3 = S3FileSystem()

    # save train_dataset to s3
    training_input_path = f's3://{sagemaker_local_session.default_bucket()}/{s3_prefix}/train'
    train_dataset.save_to_disk(training_input_path, fs=s3)

    # save test_dataset to s3
    test_input_path = f's3://{sagemaker_local_session.default_bucket()}/{s3_prefix}/test'
    test_dataset.save_to_disk(test_input_path, fs=s3)

    estimator = HuggingFace(entry_point=distrilbert_script,
                            instance_type='local_gpu',
                            sagemaker_session=sagemaker_local_session,
                            image_uri=docker_image,
                            instance_count=1,
                            role=ROLE,
                            py_version=py_version,
                            hyperparameters = hyperparameters)

    estimator.fit({'train':f's3://{sagemaker_local_session.default_bucket()}/{s3_prefix}/train', 'test':f's3://{sagemaker_local_session.default_bucket()}/{s3_prefix}/test'})
