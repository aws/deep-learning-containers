# Future
from __future__ import print_function

# Standard Library
import argparse
import math
import random

# Third Party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import StepLR
from torchnet.dataset import SplitDataset
from torchvision import datasets, transforms

# First Party
import smdistributed.modelparallel.torch as smp

# SM Distributed: import scaler from smdistributed.modelparallel.torch.amp, instead of torch.cuda.amp

# Make cudnn deterministic in order to get the same losses across runs.
# The following two lines can be removed if they cause a performance impact.
# For more details, see:
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, 1)
        return output


class GroupedNet(nn.Module):
    def __init__(self):
        super(GroupedNet, self).__init__()
        self.net1 = Net1()
        self.net2 = Net2()

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x


# SM Distributed: Define smp.step. Return any tensors needed outside.
@smp.step
def train_step(args, model, scaler, data, target):
    with autocast(args.amp > 0):
        output = model(data)

    loss = F.nll_loss(output, target, reduction="mean")

    scaled_loss = scaler.scale(loss) if args.amp else loss
    model.backward(scaled_loss)
    return output, loss


def train(args, model, scaler, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # SM Distributed: Move input tensors to the GPU ID used by the current process,
        # based on the set_device call.
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # Return value, loss_mb is a StepOutput object
        _, loss_mb = train_step(args, model, scaler, data, target)

        # SM Distributed: Average the loss across microbatches.
        loss = loss_mb.reduce_mean()

        if args.amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            # some optimizers like adadelta from PT 1.8 dont like it when optimizer.step is called with no param
            # this is a bug in PT 1.8
            if len(list(model.local_parameters())) > 0:
                optimizer.step()

        if smp.rank() == 0 and batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break
        if args.num_batches and batch_idx + 1 == args.num_batches:
            break


# SM Distributed: Define smp.step for evaluation.
@smp.step
def test_step(model, data, target):
    output = model(data)
    loss = F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    return loss, correct


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # SM Distributed: Moves input tensors to the GPU ID used by the current process
            # based on the set_device call.
            data, target = data.to(device), target.to(device)

            # Since test_step returns scalars instead of tensors,
            # test_step decorated with smp.step will return lists instead of StepOutput objects.
            loss_batch, correct_batch = test_step(model, data, target)
            test_loss += sum(loss_batch)
            correct += sum(correct_batch)
            if args.num_batches and batch_idx + 1 == args.num_batches:
                break

    test_loss /= len(test_loader.dataset)
    if smp.mp_rank() == 0:
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
    return test_loss


def get_parser():
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
        default=64,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, metavar="N", help="number of epochs to train (default: 14)"
    )
    parser.add_argument(
        "--lr", type=float, default=4.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="quickly check a single pass"
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--partial-checkpoint", type=str, default="", help="The checkpoint path to load"
    )
    parser.add_argument(
        "--full-checkpoint", type=str, default="", help="The checkpoint path to load"
    )
    parser.add_argument(
        "--save-full-model", action="store_true", default=False, help="For Saving the current Model"
    )
    parser.add_argument(
        "--save-partial-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--assert-losses", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--ddp", type=int, default=0)
    parser.add_argument('--mp_parameters', type=str, default='')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("The script requires CUDA support, but CUDA not available")
    use_ddp = args.ddp > 0
    

    # Fix seeds in order to get the same losses across runs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    smp.init()

    # SM Distributed: Set the device to the GPU ID used by the current process.
    # Input tensors should be transferred to this device.
    torch.cuda.set_device(smp.local_rank())
    device = torch.device("cuda")
    kwargs = {"batch_size": args.batch_size}
    kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": False})

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    if args.data_dir is None:
        # SM Distributed: Download only on a single process per instance.
        # When this is not present, the file is corrupted by multiple processes trying
        # to download and extract at the same time
        args.data_dir = "../data"
        if smp.local_rank() == 0:
            dataset1 = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        smp.barrier()

    dataset1 = datasets.MNIST(args.data_dir, train=True, download=False, transform=transform)

    if (use_ddp) and smp.dp_size() > 1:
        partitions_dict = {f"{i}": 1 / smp.dp_size() for i in range(smp.dp_size())}
        dataset1 = SplitDataset(dataset1, partitions=partitions_dict)
        dataset1.select(f"{smp.dp_rank()}")

    # Download and create dataloaders for train and test dataset
    dataset2 = datasets.MNIST(args.data_dir, train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = GroupedNet()

    # SMP handles the transfer of parameters to the right device
    # and the user doesn't need to call 'model.to' explicitly.
    # model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # SM Distributed: Use the DistributedModel container to provide the model
    # to be partitioned across different ranks. For the rest of the script,
    # the returned DistributedModel object should be used in place of
    # the model provided for DistributedModel class instantiation.
    model = smp.DistributedModel(model)
    scaler = smp.amp.GradScaler()
    optimizer = smp.DistributedOptimizer(optimizer)

    if args.partial_checkpoint:
        checkpoint = smp.load(args.partial_checkpoint, partial=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif args.full_checkpoint:
        checkpoint = smp.load(args.full_checkpoint, partial=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, scaler, device, train_loader, optimizer, epoch)
        test_loss = test(args, model, device, test_loader)
        scheduler.step()

    if args.save_partial_model:
        if smp.dp_rank() == 0:
            model_dict = model.local_state_dict()
            opt_dict = optimizer.local_state_dict()
            smp.save(
                {"model_state_dict": model_dict, "optimizer_state_dict": opt_dict},
                f"./pt_mnist_checkpoint.pt",
                partial=True,
            )

    if args.save_full_model:
        if smp.dp_rank() == 0:
            model_dict = model.state_dict()
            opt_dict = optimizer.state_dict()
            smp.save(
                {"model_state_dict": model_dict, "optimizer_state_dict": opt_dict},
                "./pt_mnist_checkpoint.pt",
                partial=False,
            )

    # Waiting the save checkpoint to be finished before run another allgather_object
    smp.barrier()

    if args.assert_losses:
        if use_ddp:
            # SM Distributed: If using data parallelism, gather all losses across different model
            # replicas and check if losses match.

            losses = smp.allgather(test_loss, smp.DP_GROUP)
            for l in losses:
                assert math.isclose(l, losses[0])

            assert test_loss < 0.18
        else:
            assert test_loss < 0.08

    # For CI/CD
    smp.barrier()
    print("SMP training finished successfully")


if __name__ == "__main__":
    main()
