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

# https://github.com/fastai/fastai/blob/master/nbs/examples/mnist_items.py
# imports and the code was as much preserved to match the official example

from fastai.vision.all import *

items = get_image_files(untar_data(URLs.MNIST))
splits = GrandparentSplitter(train_name='training', valid_name='testing')(items)
tds = Datasets(items, [PILImageBW.create, [parent_label, Categorize()]], splits=splits)

if __name__ == '__main__':
    data = tds.dataloaders(bs=256, after_item=[ToTensor(), IntToFloatTensor()]).cuda()
    learn = vision_learner(data, resnet18, metrics=accuracy, path='/opt/ml', model_dir='model')
    learn.fit_one_cycle(1, 1e-2)
    learn.save('model')
