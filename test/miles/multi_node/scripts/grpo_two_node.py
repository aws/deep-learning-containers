"""Upstream's short Qwen2.5-0.5B GRPO run, spread across two nodes.

Runs in the image on the Ray head node of a two-node Ray cluster that the caller started
(MILES_SCRIPT_EXTERNAL_RAY=1). Megatron trains on 8 GPUs and SGLang serves rollouts on 8
more, and each node has 8, so the two land on different nodes. Every weight update is
then an NCCL broadcast from the trainer to the engines over the network. Miles'
--ci-test mode checks that the trainer's log-probs match SGLang's and that every engine
picks up the new weights.

The arguments are those of upstream's tests/e2e/short/test_qwen2.5_0.5B_gsm8k_async_short.py
(v0.1.3), with three changes: 8 training and 8 rollout GPUs instead of 1 and 1, Ray's
object store instead of a Mooncake store bound to 127.0.0.1, and no Ray start or stop.

Usage:
    python grpo_two_node.py prepare   # on every node: download the model and data
    python grpo_two_node.py           # on the head node: run the job
"""

import os
import sys

from miles.utils.external_utils import command_utils

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
GPUS_PER_NODE = 8


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        # 3, not 2: train_async.py prefetches rollout k+1 during train k, so rollout 2 is the
        # first one generated on updated weights.
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 16 "
        "--dynamic-sampling-filter-path "
        "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 32 "
        "--pause-generation-mode in_place "
        "--namespaced-radix-cache "
    )

    eval_args = (
        "--eval-interval 8 "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.65 --sglang-enable-metrics "
    )

    ci_args = "--ci-test --sglang-enable-prefill-weight-versions --ci-assert-prefill-lag-max 3 "

    health_check_args = (
        "--rollout-health-check-interval 5 "
        "--rollout-health-check-timeout 10 "
        "--rollout-health-check-first-wait 0 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {GPUS_PER_NODE} "
        f"--rollout-num-gpus {GPUS_PER_NODE} "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{health_check_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=GPUS_PER_NODE,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    prepare()
    if sys.argv[1:] != ["prepare"]:
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(proxy_var, None)
        execute()
