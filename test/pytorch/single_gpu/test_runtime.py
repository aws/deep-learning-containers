"""Validate DataLoader multi-worker, GPU memory management, and ONNX export."""


def test_dataloader_multiworker(run_in_container):
    """Multi-worker DataLoader with pin_memory — validates /dev/shm is large enough."""
    script = (
        "import torch\n"
        "from torch.utils.data import DataLoader, TensorDataset\n"
        "ds = TensorDataset(torch.randn(256, 32), torch.randn(256, 1))\n"
        "dl = DataLoader(ds, batch_size=32, num_workers=2, pin_memory=True)\n"
        "count = 0\n"
        "for x, y in dl:\n"
        "    assert x.is_pinned()\n"
        "    count += x.shape[0]\n"
        "assert count == 256, f'Expected 256 samples, got {count}'\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True, shm_size="1g")


def test_gpu_memory_cleanup(run_in_container):
    """Allocate GPU memory, delete tensors, verify memory is freed."""
    script = (
        "import torch, gc\n"
        "torch.cuda.empty_cache()\n"
        "baseline = torch.cuda.memory_allocated()\n"
        "x = torch.randn(1024, 1024, device='cuda')\n"
        "peak = torch.cuda.memory_allocated()\n"
        "assert peak > baseline, 'Memory did not increase after allocation'\n"
        "del x\n"
        "gc.collect()\n"
        "torch.cuda.empty_cache()\n"
        "freed = torch.cuda.memory_allocated()\n"
        "assert freed <= baseline + 1024, f'Memory not freed: baseline={baseline}, after={freed}'\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True)


def test_amp_grad_scaler(run_in_container):
    """Verify AMP GradScaler training loop produces finite gradients."""
    script = (
        "import torch\n"
        "import torch.nn as nn\n"
        "m = nn.Linear(64, 1).cuda()\n"
        "opt = torch.optim.SGD(m.parameters(), lr=0.01)\n"
        "scaler = torch.amp.GradScaler()\n"
        "x = torch.randn(32, 64, device='cuda')\n"
        "y = torch.randn(32, 1, device='cuda')\n"
        "for _ in range(5):\n"
        "    opt.zero_grad()\n"
        "    with torch.amp.autocast('cuda'):\n"
        "        loss = nn.functional.mse_loss(m(x), y)\n"
        "    scaler.scale(loss).backward()\n"
        "    scaler.step(opt)\n"
        "    scaler.update()\n"
        "assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True)


def test_onnx_export(run_in_container):
    """Verify torch.onnx.export produces a valid ONNX file."""
    script = (
        "import torch, torch.nn as nn, os\n"
        "m = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)).cuda()\n"
        "x = torch.randn(1, 16, device='cuda')\n"
        "path = '/tmp/model.onnx'\n"
        "torch.onnx.export(m, x, path, input_names=['input'], output_names=['output'])\n"
        "assert os.path.getsize(path) > 0, 'ONNX file is empty'\n"
        "os.remove(path)\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True)
