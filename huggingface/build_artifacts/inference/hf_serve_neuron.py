"""Launch hf-serve with Torch NeuronX as the default device.

With Torch NeuronX 2.13+ (the PyTorch-native backend), Trainium is a regular
PyTorch device named ``neuron``. hf-serve forwards ``--device`` unchanged to
``transformers.pipeline`` and ``SentenceTransformer``, but its CLI only accepts
``auto|balanced|cuda|cpu|mps``, and ``auto`` never selects ``neuron``. This
launcher adds ``neuron`` to the accepted values and makes it the default, so
zero-code deployments (``HF_MODEL_ID`` + ``HF_TASK``) run on Neuron in eager mode.
Set ``DEVICE=cpu`` (or pass ``--device``) to override.

It also bridges an incompatibility between hf-serve 0.2.0 and Transformers 5.22
in the text-generation task (remove it once hf-serve supports Transformers 5.22):
hf-serve passes ``attn_implementation`` when it builds the pipeline and
``generate_kwargs={...}`` when it calls it. Transformers 5.22 forwards both to
``generate()``, which rejects them. ``generate_kwargs`` is flattened into plain
generation parameters (plus the tokenizer that ``stop_strings`` needs), and
``attn_implementation`` is dropped. The model has
already been loaded with its default attention (SDPA), which also avoids the
CUDA-only Hub kernel that hf-serve requests.
"""

import functools
import os

from hf_serve import cli

NEURON_DEVICE = "neuron"


def _enable_neuron_device(parser):
    for action in parser._actions:
        if action.dest != "device":
            continue
        if NEURON_DEVICE not in action.choices:
            action.choices = [*action.choices, NEURON_DEVICE]
        if not os.getenv("DEVICE"):
            action.default = NEURON_DEVICE
        return
    raise RuntimeError("hf-serve CLI has no --device argument")


def _patch_text_generation_pipeline():
    from transformers.pipelines.text_generation import TextGenerationPipeline

    sanitize_parameters = TextGenerationPipeline._sanitize_parameters

    @functools.wraps(sanitize_parameters)
    def patched_sanitize_parameters(self, *args, generate_kwargs=None, **kwargs):
        kwargs.pop("attn_implementation", None)
        kwargs = {**(generate_kwargs or {}), **kwargs}
        if kwargs.get("stop_strings") and "tokenizer" not in kwargs:
            # generate() needs the tokenizer to match stop strings.
            kwargs["tokenizer"] = self.tokenizer
        return sanitize_parameters(self, *args, **kwargs)

    TextGenerationPipeline._sanitize_parameters = patched_sanitize_parameters


def main():
    _enable_neuron_device(cli.parser)
    _patch_text_generation_pipeline()
    cli.main()


if __name__ == "__main__":
    main()
