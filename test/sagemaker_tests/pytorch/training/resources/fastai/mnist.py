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

# Based on https://github.com/fastai/fastai/blob/master/examples/train_mnist.py
# imports and the code was as much preserved to match the official example
from fastai.script import *
from fastai.vision import *

@call_parse
def main():
    tgz_path = os.environ.get('SM_CHANNEL_TRAINING')
    path = os.path.join(tgz_path, 'mnist_tiny')
    tarfile.open(f'{path}.tgz', 'r:gz').extractall(tgz_path)
    tfms = (rand_pad(2, 28), [])
    data = ImageDataBunch.from_folder(path, ds_tfms=tfms, bs=64)
    data.normalize(imagenet_stats)
    learn = create_cnn(data, models.resnet18, metrics=accuracy, path='/opt/ml', model_dir='model')
    learn.fit_one_cycle(1, 0.02)
    learn.save('model')
