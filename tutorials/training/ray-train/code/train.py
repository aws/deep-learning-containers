"""Multi-node FSDP fine-tuning on a KubeRay RayCluster running on Amazon EKS.

What this proves, and what it deliberately does not:
    This job's pass/fail bar is infrastructure-level, not training quality:
    that the Ray Train DLC's bundled EFA stack (libfabric, aws-ofi-nccl) is
    actually exercised -- NCCL selects the ``efa`` provider rather than
    silently falling back to sockets -- across both nodes, and that the job
    starts and completes. See ``deploy_ray_train_job.sh`` for the log checks
    that verify EFA is carrying the job and confirm EFA's (network-level,
    non-GPUDirect) RDMA capability via ``fi_info``. The loss values and peak
    memory logged below are informational only; this script trains on
    synthetic token ids, so there is nothing to actually learn.

Submitted via ``deploy_ray_train_job.sh``, which runs the equivalent of:
    ray job submit --address http://localhost:8265 --working-dir . \\
        -- python3 train.py --model_id Qwen/Qwen2.5-1.5B
"""

from argparse import ArgumentParser
from functools import partial
import logging
import sys

import torch
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM

import ray
import ray.train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def setup_logging():
    logger = logging.getLogger(__name__)
    if logger.handlers or hasattr(logger, "_configured"):
        return logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logger._configured = True
    logger.propagate = False
    return logger


logger = setup_logging()


def read_params():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-1.5B",
        help="Ungated causal LM. Sized so full fine-tuning needs >1 GPU's worth of memory.",
    )
    parser.add_argument("--steps", type=int, default=5, help="Optimizer steps; this is a correctness smoke test, not a full training run.")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-worker batch size. Kept small: the point is the parameter/optimizer memory, not throughput.")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Ray Train workers. 0 = auto-size from the cluster's GPU count.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")
    return args


def train_func(config):
    """Runs on every Ray Train worker. This is the body Ray fans out."""
    model_id = config["model_id"]
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    steps = config["steps"]
    lr = config["learning_rate"]

    ctx = ray.train.get_context()
    world_size, rank = ctx.get_world_size(), ctx.get_world_rank()
    device = ray.train.torch.get_device()
    logger.info(f"[train] rank={rank} world_size={world_size} device={device}")

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    vocab_size = model.config.vocab_size

    # size_based_auto_wrap_policy (rather than a model-specific transformer
    # layer class) keeps this script portable across model architectures.
    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=int(1e7))
    model = ray.train.torch.prepare_model(
        model,
        parallel_strategy="fsdp",
        parallel_strategy_kwargs={"auto_wrap_policy": auto_wrap_policy},
    )

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    torch.cuda.reset_peak_memory_stats(device)

    # Synthetic token ids: the point is exercising real FSDP collectives
    # across nodes, not achieving a training result, so there is no dataset.
    for step in range(steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=input_ids).loss  # FSDP all-gathers each unit's parameters across nodes.
        loss.backward()  # FSDP re-gathers parameters and reduce-scatters gradients across nodes, over EFA.
        optimizer.step()  # Updates only this rank's shard; no cross-node traffic.

        peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        ray.train.report({"step": step, "loss": loss.item(), "peak_gpu_mem_gb": round(peak_mem_gb, 2)})
        logger.info(f"[train] rank={rank} step={step} loss={loss.item():.4f} peak_gpu_mem_gb={peak_mem_gb:.2f}")

    logger.info(f"[train] SUCCESS: rank={rank}/{world_size} completed {steps} step(s).")


def resolve_num_workers(requested):
    """Auto-size Ray Train workers to the cluster's GPU count."""
    if requested and requested > 0:
        return requested
    gpus = int(ray.cluster_resources().get("GPU", 0))
    if gpus < 1:
        logger.error(f"No GPUs visible in the cluster: {ray.cluster_resources()}")
        sys.exit(1)
    return gpus


if __name__ == "__main__":
    # `ray job submit` runs this inside a job supervisor actor that is already
    # part of the cluster; address="auto" attaches to it instead of starting a
    # new local Ray instance.
    ray.init(address="auto")

    args = read_params()
    logger.info(f"Training arguments: {args}")
    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")

    num_workers = resolve_num_workers(args.num_workers)
    logger.info(f"Ray Train: num_workers={num_workers} (FSDP-sharded across all of them)")

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "model_id": args.model_id,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
        },
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True),
        run_config=RunConfig(name="ray-train-fsdp-llm"),
    )

    result = trainer.fit()
    logger.info(f"Training complete. Final metrics: {result.metrics}")
