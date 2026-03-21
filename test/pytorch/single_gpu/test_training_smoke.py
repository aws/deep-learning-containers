"""Smoke test: train a small MLP, verify loss decreases, checkpoint round-trip."""


def test_training_loss_decreases(run_in_container):
    script = (
        "import torch\n"
        "import torch.nn as nn\n"
        "m = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).cuda()\n"
        "opt = torch.optim.Adam(m.parameters())\n"
        "x = torch.randn(128, 32, device='cuda')\n"
        "y = torch.randn(128, 1, device='cuda')\n"
        "first = None\n"
        "for i in range(10):\n"
        "    loss = nn.functional.mse_loss(m(x), y)\n"
        "    if i == 0: first = loss.item()\n"
        "    opt.zero_grad(); loss.backward(); opt.step()\n"
        "assert loss.item() < first\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True)


def test_checkpoint_roundtrip(run_in_container):
    script = (
        "import torch, torch.nn as nn, os\n"
        "m = nn.Linear(16, 16).cuda()\n"
        "path = '/tmp/ckpt.pt'\n"
        "torch.save(m.state_dict(), path)\n"
        "m2 = nn.Linear(16, 16).cuda()\n"
        "m2.load_state_dict(torch.load(path, weights_only=True))\n"
        "for p1, p2 in zip(m.parameters(), m2.parameters()):\n"
        "    assert torch.equal(p1, p2)\n"
        "os.remove(path)\n"
        "print('ok')"
    )
    run_in_container(f'python -c "{script}"', gpu=True)
