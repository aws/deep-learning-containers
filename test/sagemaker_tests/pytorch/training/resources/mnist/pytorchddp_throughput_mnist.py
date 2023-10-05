# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
from __future__ import print_function
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# import smdistributed.dataparallel.torch.torch_smddp
from torch.distributed import Backend

"""
For EC2 conda env, in controller node run 'source /shared/pytorch_env/bin/activate', then launch the training using:
/opt/amazon/openmpi/bin/mpirun --hostfile /shared/hostfile -N 1 --tag-output --bind-to none --oversubscribe \
--allow-run-as-root --mca btl_tcp_if_exclude lo,docker0 -x PATH -x LD_LIBRARY_PATH -x RDMAV_FORK_SAFE=1 \
-x NCCL_DEBUG=INFO -x FI_EFA_USE_DEVICE_RDMA=1 sh -c ' /shared/pytorch_env/bin/python \
-m torch.distributed.launch --nproc_per_node=8 --nnodes=`wc -l < /shared/hostname` --node_rank=$OMPI_COMM_WORLD_RANK \
--master_addr=`head -n 1 /shared/hostfile` --master_port=12358 /shared/ddp_mnist.py --data_loader_workers=8'
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, rank):
    model.train()
    train_loader.sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and rank == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data) * args.world_size,
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
        if args.verbose:
            print("Batch", batch_idx, "from rank", rank)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
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
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=os.getenv("LOCAL_RANK", -1),
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="For displaying smdistributed.dataparallel-specific logs",
    )
    parser.add_argument(
        "--data_loader_workers",
        type=int,
        default=0,
        help="For displaying smdistributed.dataparallel-specific logs",
    )
    parser.add_argument(
        "--data-path", type=str, default="/tmp/data", help="Path for downloading the MNIST dataset"
    )
    parser.add_argument("--inductor", type=int, default=0, help="pytorch with inductor")
    parser.add_argument("--use_mpi", type=str, default=False, help="run pytorch using mpi")
    
    args = parser.parse_args()

    if use_mpi:
        if not os.getenv("WORLD_SIZE"):
            os.environ["WORLD_SIZE"] = str(os.getenv("OMPI_COMM_WORLD_SIZE"))

        if not os.getenv("RANK"):
            os.environ["RANK"] = str(os.getenv("OMPI_COMM_WORLD_RANK"))

        if not os.getenv("MASTER_ADDR"):
            os.environ["MASTER_ADDR"] = os.environ["SM_HOSTS"][0]
            os.environ["MASTER_PORT"] = "55555"
            
        if not local_rank:
            args.local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK"))

    args.world_size = int(os.environ["WORLD_SIZE"])

    args.lr = 1.0
    args.batch_size //= args.world_size // 8
    args.batch_size = max(args.batch_size, 1)
    data_path = args.data_path
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = args.local_rank

    torch.cuda.set_device(local_rank)
    
    if args.verbose:
        print(
            "Hello from rank",
            rank,
            "of local_rank",
            local_rank,
            "in world size of",
            args.world_size,
        )
    if not torch.cuda.is_available():
        raise Exception(
            "Must run smdistributed.dataparallel MNIST example on CUDA-capable devices."
        )
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{local_rank}")
    # select a single rank per node to download data
    is_first_local_rank = local_rank == 0
    if is_first_local_rank:
        train_dataset = datasets.MNIST(
            data_path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
    dist.barrier()  # prevent other ranks from accessing the data early
    if not is_first_local_rank:
        train_dataset = datasets.MNIST(
            data_path,
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.data_loader_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    if rank == 0:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                data_path,
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=args.test_batch_size,
            shuffle=True,
        )

    model = Net()

    use_inductor = args.inductor == 1
    if use_inductor:
        model = torch.compile(model, backend="inductor", mode="default")

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, rank)
        if rank == 0:
            test(model, device, test_loader)
        scheduler.step()
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
