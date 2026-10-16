"""Verify SGLang, the rollout engine Miles drives, serves a real model on the GPU.

The test workflow mounts a small model and sets MILES_TEST_MODEL_DIR to it.
"""

import os

import pytest
import torch

PROMPT = "Q: What is the capital of Germany?\nA: Berlin\nQ: What is the capital of France?\nA:"
GREEDY = {"temperature": 0.0, "max_new_tokens": 8}


@pytest.fixture(scope="module")
def engine():
    import sglang

    model_dir = os.environ.get("MILES_TEST_MODEL_DIR")
    assert model_dir, "MILES_TEST_MODEL_DIR must point at a Hugging Face model directory"
    # Leave the memory the earlier tests in this process cached to the engine.
    torch.cuda.empty_cache()
    engine = sglang.Engine(
        model_path=model_dir,
        dtype="bfloat16",
        context_length=4096,
        mem_fraction_static=0.6,
        random_seed=0,
    )
    yield engine
    engine.shutdown()


def test_greedy_completion_is_correct(engine):
    assert "Paris" in engine.generate(PROMPT, GREEDY)["text"]


def test_completion_returns_token_logprobs(engine):
    # Miles trains on the rollout's token log-probs, so the engine must return one per token.
    output = engine.generate(PROMPT, GREEDY, return_logprob=True)
    meta = output["meta_info"]
    logprobs = [entry[0] for entry in meta["output_token_logprobs"]]
    assert len(logprobs) == meta["completion_tokens"] > 0
    assert all(logprob <= 0.0 for logprob in logprobs)
