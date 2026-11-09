"""SGLang, the rollout engine Miles drives, serves a model split across GPUs.

MILES_TEST_MODEL_DIR points at a Hugging Face model directory.
"""

import argparse
import os

import sglang

SUCCESS_MARKER = "MILES_MULTI_GPU_OK"
PROMPT = "Q: What is the capital of Germany?\nA: Berlin\nQ: What is the capital of France?\nA:"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    args = parser.parse_args()

    engine = sglang.Engine(
        model_path=os.environ["MILES_TEST_MODEL_DIR"],
        tp_size=args.tp,
        dtype="bfloat16",
        context_length=4096,
        mem_fraction_static=0.7,
        random_seed=0,
    )
    try:
        text = engine.generate(PROMPT, {"temperature": 0.0, "max_new_tokens": 8})["text"]
    finally:
        engine.shutdown()
    assert "Paris" in text, text
    print(SUCCESS_MARKER)


if __name__ == "__main__":
    main()
