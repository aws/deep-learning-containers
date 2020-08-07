import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP

model = torchvision.models.resnet50()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def basic_resnet(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    model = torchvision.models.resnet50().to(torch.cuda.current_device())
    ddp_model = DDP(model, device_ids=[torch.cuda.current_device()])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # Creates a GradScaler once at the beginning of training
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1):
        with torch.cuda.amp.autocast():
            _inputs = torch.randn(20, 3, 32, 32).to(torch.cuda.current_device())
            outputs = ddp_model(_inputs)
            labels = torch.randn(1000).to(torch.cuda.current_device())
            loss = loss_fn(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"Done with epoch {epoch}")


def run_test(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    run_test(basic_resnet, world_size=2)
