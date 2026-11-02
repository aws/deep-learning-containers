"""`strands-decider serve` with two additions for SageMaker, set by environment variables:

DECIDER_WARMUP      "1" (default) answers requests after the model loads and before the server listens, so no
                    caller waits for a compile: about 37 s on the first request for the Qwen3.5 models, then about
                    0.8 s the first time each number of questions (1 to --max-batch) is asked. /ping fails until
                    then, which keeps SageMaker from routing to a cold container. "0" skips it.
DECIDER_DEVICE_MAP  unset (default) keeps the model on one GPU, as upstream does. "auto" (or another transformers
                    device_map) splits it across the visible GPUs, for a model larger than one GPU, such as the
                    26B-A4B on L40S. The layers run one GPU after the other, so it adds memory, not speed.

All `strands-decider serve` options still apply; this module only wraps upstream functions before the CLI runs.
"""

import os
import sys
import time

import torch
from strands_decider import cli, server
from strands_decider.modeling import StrandsDeciderModel
from strands_decider.schema import SystemOneRequest

# The question types and paths a request can take: one question runs the batched forward, several run the
# shared-prefix forward; noul, choice and score read different slots. Then every question count up to --max-batch
# (larger requests run in chunks of at most that many).
WARMUP_REQUESTS = [
    {
        "state": "Help! My payouts have been failing for 3 days and nobody answers my emails.",
        "questions": {
            "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
            "department": {
                "type": "choice",
                "instructions": "Route it.",
                "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"},
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
        },
    },
    {
        "state": "The order arrived on time and works as described.",
        "questions": {
            "is_positive": {"type": "noul", "instructions": "Is the customer satisfied?"}
        },
    },
]


def log(message: str) -> None:
    print(f"[sagemaker-serve] {message}", file=sys.stderr, flush=True)


def warm_up_after_load() -> None:
    create_app = server.create_app

    def create_and_warm_up(*args, **kwargs):
        app = create_app(*args, **kwargs)
        engine = server.get_engine()
        for request in WARMUP_REQUESTS:
            started = time.perf_counter()
            engine.evaluate(SystemOneRequest.model_validate(request))
            log(
                f"warm-up request ({len(request['questions'])} questions): {time.perf_counter() - started:.1f} s"
            )
        started = time.perf_counter()
        max_batch = kwargs.get("max_batch", 32)
        for n in range(2, max_batch + 1):
            questions = {
                f"q{i}": {"type": "noul", "instructions": f"Is point {i} about money?"}
                for i in range(n)
            }
            engine.evaluate(
                SystemOneRequest.model_validate(
                    {"state": WARMUP_REQUESTS[0]["state"], "questions": questions}
                )
            )
        log(f"warm-up of 2 to {max_batch} questions: {time.perf_counter() - started:.1f} s")
        return app

    server.create_app = create_and_warm_up


def load_across_gpus(device_map: str) -> None:
    load = StrandsDeciderModel.load
    first_gpu = torch.device("cuda", 0)

    def load_split(path, **kwargs):
        model = load(path, device_map=device_map, **kwargs)
        log(f"model on {sorted({str(p.device) for p in model.torso.parameters()})}")
        # The engine creates its inputs on cuda:0 and moves the model there with .to(): keep the split, put the
        # head on cuda:0, and hand it the decoder's output from whichever GPU holds the last layer.
        model.head.to(first_gpu)
        model.to = lambda *_, **__: model

        def output_to_first_gpu(_module, _args, output):
            output.last_hidden_state = output.last_hidden_state.to(first_gpu)
            return output

        model.torso.get_base_model().register_forward_hook(output_to_first_gpu)
        return model

    StrandsDeciderModel.load = load_split


if __name__ == "__main__":
    if os.environ.get("DECIDER_DEVICE_MAP"):
        load_across_gpus(os.environ["DECIDER_DEVICE_MAP"])
    if os.environ.get("DECIDER_WARMUP", "1") != "0":
        warm_up_after_load()
    cli.main()
