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
import argparse
import logging
import os
import sys

import cv2 as cv
import sagemaker_containers
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
from smdebug.pytorch import *
import numpy as np
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
class Net(nn.Module):
    def __init__(self):
        logger.info("Create neural network module")

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def parse_args():
    env = sagemaker_containers.training_env()
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")

    parser.add_argument('--data_dir', type=str)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of Epochs")
    parser.add_argument(
        "--smdebug_path",
        type=str,
        default=None,
        help="S3 URI of the bucket where tensor data will be stored.",
    )
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--random_seed", type=bool, default=False)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Reduce the number of training "
             "and evaluation steps to the give number if desired."
             "If this is not passed, trains for one epoch "
             "of training and validation data",
    )
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    opt = parser.parse_args()
    return opt


def _get_train_data_loader(batch_size, training_dir):
    logger.info("Get train data loader")
    dataset = datasets.MNIST(training_dir, train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=4)


def _get_test_data_loader(test_batch_size, training_dir):
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.MNIST(training_dir, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=False, num_workers=4)


def create_smdebug_hook(output_s3_uri):
    # With the following SaveConfig, we will save tensors for steps 1, 2 and 3
    # (indexing starts with 0).
    save_config = SaveConfig(save_steps=[1, 2, 3])

    # Create a hook that logs weights, biases and gradients while training the model.
    hook = Hook(
        out_dir=output_s3_uri,
        save_config=save_config,
        include_collections=["weights", "gradients", "losses"],
    )
    return hook


def train(model, device, optimizer, hook, epochs, log_interval, training_dir):
    criterion = nn.CrossEntropyLoss()
    hook.register_loss(criterion)

    trainloader = _get_train_data_loader(4, training_dir)
    validloader = _get_test_data_loader(4, training_dir)

    for epoch in range(epochs):
        model.train()
        hook.set_mode(modes.TRAIN)
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(data), len(trainloader.sampler),
                           100. * i / len(trainloader), loss.item()))

        test(model, hook, validloader, device, criterion)


def test(model, hook, test_loader, device, loss_fn):
    model.eval()
    hook.set_mode(modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.debug('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    opt = parse_args()

    if opt.random_seed:
        torch.manual_seed(128)
        random.seed(12)
        np.random.seed(2)

    device = torch.device("cpu")
    out_dir = opt.smdebug_path
    training_dir = opt.data_dir
    hook = create_smdebug_hook(out_dir)
    model = Net().to(device)
    hook.register_hook(model)
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum)
    train(model, device, optimizer, hook, opt.epochs, opt.log_interval, training_dir)
    print("Training is complete")

    from smdebug.trials import create_trial
    print("Created the trial with out_dir {0}".format(out_dir))
    trial = create_trial(out_dir)
    assert trial
    print("Train steps: " + str(trial.steps(mode=modes.TRAIN)))
    print("Eval steps: " + str(trial.steps(mode=modes.EVAL)))

    print(
        f"trial.tensor_names() = {trial.tensor_names()}"
    )  
    # if loss collection tensors not in in trial.tensor_names()
    # means they were not saved

    print(f"collection_manager = {hook.collection_manager}")

    weights_tensors = hook.collection_manager.get("weights").tensor_names
    print(f"'weights' collection tensors = {weights_tensors}")
    assert len(weights_tensors) > 0

    gradients_tensors = hook.collection_manager.get("gradients").tensor_names
    print(f"'gradients' collection tensors = {gradients_tensors}")
    assert len(gradients_tensors) > 0

    losses_tensors = hook.collection_manager.get("losses").tensor_names
    print(f"'losses' collection tensors = {losses_tensors}")
    assert len(losses_tensors) > 0

    assert all(
        [name in trial.tensor_names() for name in losses_tensors]
    )

    print("Validation Complete")

if __name__ == "__main__":
    main()

