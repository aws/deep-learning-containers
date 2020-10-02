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

# Based on https://github.com/fastai/fastai/blob/master/examples/train_cifar.py
# imports and the code was as much preserved to match the official example
from fastai.script import *
from fastai.vision import *
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
import torch

torch.backends.cudnn.benchmark = True

@call_parse
def main(gpu:Param("GPU to run on", str)=None):
    """Distrubuted training of CIFAR-10.
    Fastest speed is if you run as follows:
    python -m fastai.launch cifar.py"""
    gpu = setup_distrib(gpu)
    n_gpus = int(os.environ.get("WORLD_SIZE", 1))
    tgz_path = os.environ.get('SM_CHANNEL_TRAINING')
    path = os.path.join(tgz_path, 'cifar10_tiny')
    tarfile.open(f'{path}.tgz', 'r:gz').extractall(tgz_path)
    ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
    workers = min(16, num_cpus()//n_gpus)
    data = ImageDataBunch.from_folder(path, valid='test', ds_tfms=ds_tfms, bs=512//n_gpus,
                                      num_workers=workers)
    data.normalize(cifar_stats)
    learn = Learner(data, wrn_22(), metrics=accuracy, path='/opt/ml', model_dir='model')
    if gpu is None:
        learn.model = nn.DataParallel(learn.model)
    else:
        learn.to_distributed(gpu)
    learn.to_fp16()
    learn.fit_one_cycle(2, 3e-3, wd=0.4)
    learn.save('model')
