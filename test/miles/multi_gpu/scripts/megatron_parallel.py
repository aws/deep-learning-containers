"""Run under torchrun: Megatron-Core trains a small GPT with tensor and data parallelism.

Tensor parallelism splits each layer across a pair of GPUs. Data parallelism gives each
pair its own batch and syncs gradients through Megatron's distributed optimizer, the
path Miles trains with. The loss must fall, and replicas holding the same tensor-
parallel shard must end with identical weights.
"""

import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module

SUCCESS_MARKER = "MILES_MULTI_GPU_OK"
TENSOR_PARALLEL = 2
VOCAB_SIZE = 512
SEQ_LENGTH = 128
BATCH_SIZE = 4
STEPS = 50


def build_model():
    torch.manual_seed(0)
    model_parallel_cuda_manual_seed(0)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        ffn_hidden_size=1024,
        num_attention_heads=4,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=TENSOR_PARALLEL,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LENGTH,
    ).cuda()
    model = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        Float16Module(config, model),
    )
    model.broadcast_params()
    return model


def make_batch(data_parallel_rank):
    # Each next token is a fixed function of the current one; replicas start elsewhere.
    steps = torch.arange(SEQ_LENGTH + 1, device="cuda")
    starts = (torch.arange(BATCH_SIZE, device="cuda") + data_parallel_rank * BATCH_SIZE) * 13
    tokens = (starts[:, None] + steps * 7) % VOCAB_SIZE
    positions = torch.arange(SEQ_LENGTH, device="cuda").expand(BATCH_SIZE, -1)
    # Megatron masks the positions where the mask is True.
    mask = torch.ones(SEQ_LENGTH, SEQ_LENGTH, dtype=torch.bool, device="cuda").triu(1)
    return tokens[:, :-1], positions, mask[None, None], tokens[:, 1:]


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=TENSOR_PARALLEL)

    model = build_model()
    optimizer = get_megatron_optimizer(
        OptimizerConfig(optimizer="adam", lr=1e-3, bf16=True, use_distributed_optimizer=True),
        [model],
    )
    inputs, positions, mask, labels = make_batch(parallel_state.get_data_parallel_rank())
    losses = []
    for _ in range(STEPS):
        model.zero_grad_buffer()
        optimizer.zero_grad()
        loss = model(inputs, positions, mask, labels=labels).float().mean()
        loss.backward()
        finalize_model_grads([model])
        update_successful, _, _ = optimizer.step()
        assert update_successful
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < 0.5 * losses[0], losses

    checksum = torch.stack([param.detach().double().sum() for param in model.parameters()]).sum()
    replicas = [
        torch.zeros_like(checksum) for _ in range(parallel_state.get_data_parallel_world_size())
    ]
    dist.all_gather(replicas, checksum, group=parallel_state.get_data_parallel_group())
    assert all(torch.equal(replica, checksum) for replica in replicas), replicas

    dist.barrier()
    if dist.get_rank() == 0:
        print(SUCCESS_MARKER)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
