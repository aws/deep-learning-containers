"""Train a tiny Transformers model on two Neuron ranks using TP or FSDP.

Inspired by training_scripts/finetune_overfit.py. Run on Trainium with:
    HF_DEACTIVATE_ASYNC_LOAD=1 torchrun --standalone --nproc_per_node=2 \
        transformers_neuron_training.py --parallel-mode tp
Repeat with --parallel-mode fsdp to exercise FSDP2.
"""

import argparse
import math
import os
from pathlib import Path

import torch
from datasets import Dataset
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM, Qwen3Config, Trainer, TrainingArguments, set_seed
from transformers.distributed import DistributedConfig
from transformers.distributed.utils import _distributed_barrier, _ensure_torch_distributed
from transformers.utils import is_torch_neuron_available


def main(parallel_mode):
    import torch_neuronx  # noqa: F401

    if not is_torch_neuron_available(check_device=True):
        raise RuntimeError("This training test requires a working Neuron device")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise ValueError("Launch this test with torchrun --nproc_per_node=2")

    _ensure_torch_distributed(device_type="neuron")
    try:
        set_seed(42)
        root = Path(f"/tmp/hf-neuron-{parallel_mode}")
        checkpoint = root / "initial-model"
        if torch.distributed.get_rank() == 0:
            config = Qwen3Config(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                head_dim=16,
                max_position_embeddings=64,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                use_cache=False,
            )
            AutoModelForCausalLM.from_config(config).save_pretrained(checkpoint)
        _distributed_barrier()

        tp_size = world_size if parallel_mode == "tp" else 1
        fsdp_size = world_size if parallel_mode == "fsdp" else 1
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            distributed_config=DistributedConfig(tp_size=tp_size, fsdp_size=fsdp_size),
        )
        assert model.tp_size == tp_size, "Tensor parallel degree does not match the request"
        assert model.fsdp_size == fsdp_size, "FSDP degree does not match the request"
        assert any(
            isinstance(parameter, DTensor)
            and parameter.device_mesh.size() == world_size
            and any(placement.is_shard() for placement in parameter.placements)
            for parameter in model.parameters()
        ), "The model has no parameters sharded across the requested ranks"

        tokens = [4, 5, 6, 7] * 8
        dataset = Dataset.from_dict({"input_ids": [tokens] * 16, "labels": [tokens] * 16})
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(root / "output"),
                per_device_train_batch_size=2,
                max_steps=12,
                learning_rate=1e-2,
                lr_scheduler_type="constant",
                bf16=True,
                optim="adamw_torch",
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                disable_tqdm=True,
            ),
            train_dataset=dataset,
        )
        assert trainer.args.device.type == "neuron", "Training must not fall back to CPU"
        before = {
            name: (parameter.to_local() if isinstance(parameter, DTensor) else parameter)
            .detach()
            .cpu()
            .clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        trainer.train()
        losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
        assert len(losses) == 12, f"Expected 12 logged training steps, got {losses}"
        assert all(math.isfinite(loss) for loss in losses), f"Non-finite training loss: {losses}"
        assert losses[-1] < losses[0], f"Training loss did not decrease: {losses}"
        assert any(
            not torch.equal(
                (parameter.to_local() if isinstance(parameter, DTensor) else parameter)
                .detach()
                .cpu(),
                before[name],
            )
            for name, parameter in model.named_parameters()
            if name in before
        ), "Training did not update any local parameter shard"
        print(
            f"PASS mode={parallel_mode} rank={torch.distributed.get_rank()} "
            f"tp={tp_size} fsdp={fsdp_size} loss={losses[0]:.4f}->{losses[-1]:.4f}",
            flush=True,
        )
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel-mode", choices=("tp", "fsdp"), required=True)
    main(parser.parse_args().parallel_mode)
