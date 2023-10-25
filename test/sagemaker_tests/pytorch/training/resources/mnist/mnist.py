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

# Workaround for https://github.com/pytorch/vision/issues/1938
from __future__ import print_function, absolute_import
from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)
from packaging.version import Version

import argparse
import logging
import os
import sys

import cv2 as cv
import sagemaker_training.environment
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

import torchvision
from torchvision import datasets, transforms

# from torchvision 0.9.1, 2 candidate mirror website links will be added before "resources" items automatically
# Reference PR: https://github.com/pytorch/vision/pull/3559
TORCHVISION_VERSION = "0.9.1"
if Version(torchvision.__version__) < Version(TORCHVISION_VERSION):
    datasets.MNIST.resources = [
        (
            "https://dlinfra-mnist-dataset.s3-us-west-2.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "https://dlinfra-mnist-dataset.s3-us-west-2.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "https://dlinfra-mnist-dataset.s3-us-west-2.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "https://dlinfra-mnist-dataset.s3-us-west-2.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ]

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


def _get_train_data_loader(batch_size, training_dir, is_distributed, **kwargs):
    logger.info("Get train data loader")
    dataset = datasets.MNIST(
        training_dir,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **kwargs,
    )


def _get_test_data_loader(test_batch_size, training_dir, **kwargs):
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.MNIST(
            training_dir,
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def train(args):
    is_distributed = args.backend is not None
    use_cuda = (args.processor == "gpu") or (args.num_gpus > 0)
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")
    use_inductor = args.inductor == 1
    use_mpi = args.use_mpi

    if is_distributed:
        # Initialize the distributed environment.
        if not os.getenv("MASTER_ADDR"):
            os.environ["MASTER_ADDR"] = os.environ["SM_HOSTS"][0]
            os.environ["MASTER_PORT"] = "55555"
            args.hosts = os.environ["SM_HOSTS"]
        if not os.getenv("RANK") and args.backend == "nccl":
            os.environ["RANK"] = str(os.getenv("OMPI_COMM_WORLD_RANK"))

        if not os.getenv("WORLD_SIZE") and args.backend == "nccl":
            os.environ["WORLD_SIZE"] = str(os.getenv("OMPI_COMM_WORLD_SIZE"))

        if not os.getenv("RANK"):  # for local dist job
            os.environ["RANK"] = str(args.hosts.index(args.current_host))
        if not os.getenv("WORLD_SIZE"):  # for local dist job
            os.environ["WORLD_SIZE"] = str(len(args.hosts))

        dist.init_process_group(backend=args.backend)
        logger.info(
            "Initialized the distributed environment: '{}' backend on {} processes. ".format(
                args.backend, dist.get_world_size()
            )
            + "Current rank is {}, Current host is: {}, Number of gpus: {}, device used: {}".format(
                dist.get_rank(), args.current_host, args.num_gpus, device
            )
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    # TODO: assert the logs when we move to the SDK local mode
    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    model = Net()

    if is_distributed:
        if use_cuda:
            device_id = dist.get_rank() % torch.cuda.device_count()
            device = torch.device(f"cuda:{device_id}")
            # multi-machine multi-gpu case
            logger.debug("Multi-machine multi-gpu cuda: using DistributedDataParallel.")
        # for multiprocessing distributed, the DDP constructor should always set
        # the single device scope. otherwise, DDP will use all available devices.
        for name, param in model.named_parameters():
            print(f"{dist.get_rank()} model parameters {name}: {param.size()}")
        model = torch.nn.parallel.DistributedDataParallel(model.to(device))
    else:
        model = model.to(device)

    if use_inductor:
        logger.debug("Inductor: using Inductor.")
        model = torch.compile(model, backend="inductor", mode="default")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            if is_distributed and use_cuda:
                # multi-machine multi-gpu case - allow asynchrous GPU copies of the data
                data, target = data.to(device, non_blocking=True), target.to(
                    device, non_blocking=True
                )
            else:
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.debug(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader, device)
    save_model(model, args.model_dir, args)

    if len(args.hosts) == 1 or os.environ["RANK"] == 0:
        assert_can_track_sagemaker_experiments()


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=None).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.debug(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def save_model(model, model_dir, args):
    model_pth = f"model_{args.hosts.index(args.current_host)}.pth"
    logger.info(f"Saving the model: {model_pth}")
    path = os.path.join(model_dir, model_pth)
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)


def assert_can_track_sagemaker_experiments():
    in_sagemaker_training = "TRAINING_JOB_ARN" in os.environ
    in_python_three = sys.version_info[0] == 3

    if in_sagemaker_training and in_python_three:
        import smexperiments.tracker

        with smexperiments.tracker.Tracker.load() as tracker:
            tracker.log_parameter("param", 1)
            tracker.log_metric("metric", 1.0)


if __name__ == "__main__":
    print("sys.argv: ", sys.argv)
    # test opencv
    print(cv.__version__)

    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10000,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend", type=str, default=None, help="backend for distributed training"
    )
    parser.add_argument(
        "--processor", type=str, default="cpu", help="backend for distributed training"
    )
    parser.add_argument("--inductor", type=int, default=0, help="pytorch with inductor")
    parser.add_argument("--use_mpi", type=str, default=False, help="run pytorch using mpi")

    # Container environment
    env = sagemaker_training.environment.Environment()
    parser.add_argument("--hosts", type=list, default=env.hosts)
    parser.add_argument("--current-host", type=str, default=env.current_host)
    parser.add_argument("--model-dir", type=str, default=env.model_dir)
    parser.add_argument("--data-dir", type=str, default=env.channel_input_dirs["training"])
    parser.add_argument("--num-gpus", type=int, default=env.num_gpus)

    args = parser.parse_args()

    train(args)
