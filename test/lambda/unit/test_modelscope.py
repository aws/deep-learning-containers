"""modelscope ships in the vLLM image so models can be pulled from the ModelScope hub.

Set VLLM_USE_MODELSCOPE=true and point MODEL_ID at a ModelScope model id; vLLM then
resolves the repo through modelscope instead of Hugging Face. The sglang image gets
the same package transitively via sglang[all].

This unit test guards the cheap contract: the package is installed, importable, and
still below the 1.38 cap that vLLM's repo_utils.py requires.
"""

import importlib
import importlib.metadata as md


def test_modelscope_importable():
    importlib.import_module("modelscope")


def test_modelscope_below_incompatible_release():
    # 1.38 dropped the `revision` kwarg vLLM passes; unpin after vllm#47325 lands.
    major, minor = (int(p) for p in md.version("modelscope").split(".")[:2])
    assert (major, minor) < (1, 38)
