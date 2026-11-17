"""Verify Megatron-Core with TransformerEngine layers, the stack Miles trains with, learns
on one GPU and round-trips its distributed checkpoint."""

import socket

import pytest
import torch
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module

VOCAB_SIZE = 512
SEQ_LENGTH = 128
BATCH_SIZE = 4


@pytest.fixture(scope="module", autouse=True)
def model_parallel():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
    )
    parallel_state.initialize_model_parallel(1, 1)
    yield
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()


def build_model(seed):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        ffn_hidden_size=1024,
        num_attention_heads=4,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LENGTH,
    ).cuda()
    # Megatron trains bf16 models through Float16Module, which casts every parameter,
    # the learned position embedding included, to bf16.
    return Float16Module(config, model)


def make_batch():
    # Each next token is a fixed function of the current one, so the model can learn it.
    steps = torch.arange(SEQ_LENGTH + 1, device="cuda")
    starts = torch.arange(BATCH_SIZE, device="cuda")[:, None] * 13
    tokens = (starts + steps * 7) % VOCAB_SIZE
    positions = torch.arange(SEQ_LENGTH, device="cuda").expand(BATCH_SIZE, -1)
    # Megatron masks the positions where the mask is True.
    mask = torch.ones(SEQ_LENGTH, SEQ_LENGTH, dtype=torch.bool, device="cuda").triu(1)
    return tokens[:, :-1], positions, mask[None, None], tokens[:, 1:]


def lm_loss(model, batch):
    inputs, positions, mask, labels = batch
    return model(inputs, positions, mask, labels=labels).float().mean()


def test_gpt_loss_decreases_while_training():
    model = build_model(seed=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = make_batch()
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        loss = lm_loss(model, batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < 0.5 * losses[0], losses


def test_distributed_checkpoint_round_trips(tmp_path):
    model = build_model(seed=2)
    dist_checkpointing.save(model.sharded_state_dict(prefix=""), str(tmp_path))

    restored = build_model(seed=3)
    state = dist_checkpointing.load(restored.sharded_state_dict(prefix=""), str(tmp_path))
    restored.load_state_dict(state)

    for name, param in model.state_dict().items():
        if isinstance(param, torch.Tensor):
            assert torch.equal(param, restored.state_dict()[name]), name
    model.eval()
    restored.eval()
    batch = make_batch()
    with torch.no_grad():
        assert torch.equal(lm_loss(model, batch), lm_loss(restored, batch))
