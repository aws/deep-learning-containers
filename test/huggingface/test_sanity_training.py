#!/usr/bin/env python3
"""CPU sanity tests for Hugging Face GPU and Torch NeuronX training DLCs.

Run from the repository root with the matching image config:
    docker run --rm -v "$(pwd):/workdir" --workdir /workdir \
        -e HF_IMAGE_CONFIG=<image-config.yml> \
        --entrypoint python3 <image> \
        test/huggingface/test_sanity_training.py

GPU images may need /usr/local/cuda/compat on LD_LIBRARY_PATH for imports
on hosts without GPU drivers. Models and tokenizers are built locally;
these tests do not compile or train on accelerator hardware.

HF_IMAGE_CONFIG supplies version pins and metadata.device_type (gpu or
neuronx). Without it, version and image-specific checks are skipped.
"""

import importlib
import importlib.metadata
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

# build.* keys in the image config that are not Python package pins.
NON_PACKAGE_KEYS = {"python_version", "cuda_version", "dlc_major_version", "dlc_minor_version"}

# Tiny vocabulary for the hermetic tokenizer; every word in the datasets below is in it.
WORDS = (
    "the quick brown fox jumps over lazy dog a cat sat on mat hello world "
    "what is two plus answer four yes no"
).split()


def _image_config():
    path = os.environ.get("HF_IMAGE_CONFIG")
    if not path:
        return None
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


IMAGE_CONFIG = _image_config()
DEVICE_TYPE = IMAGE_CONFIG["metadata"]["device_type"] if IMAGE_CONFIG else None
ENTRYPOINT = "/usr/local/bin/sagemaker_entrypoint.sh"


def _expected_pins():
    """Read package and framework versions from the selected image config."""
    if IMAGE_CONFIG is None:
        return None
    pins = {
        key[: -len("_version")].replace("_", "-"): str(value)
        for key, value in IMAGE_CONFIG["build"].items()
        if key.endswith("_version") and key not in NON_PACKAGE_KEYS
    }
    return pins, str(IMAGE_CONFIG["metadata"]["framework_version"])


def _build_tiny_model(path):
    """Save a randomly initialised 2-layer Qwen2 model to `path`; return its word-level tokenizer.

    The tokenizer is passed explicitly rather than saved next to the model:
    AutoTokenizer resolves a Qwen2 checkpoint to Qwen2Tokenizer, whose
    byte-level pre-tokenizer does not fit this vocabulary.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    special = ["<pad>", "<eos>", "<unk>"]
    vocab = {tok: i for i, tok in enumerate(special + WORDS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )

    config = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=vocab["<pad>"],
        eos_token_id=vocab["<eos>"],
        bos_token_id=None,
    )
    Qwen2ForCausalLM(config).save_pretrained(path)
    return tokenizer


class TestPinnedVersions(unittest.TestCase):
    """Installed packages must match the pins in the image config."""

    def setUp(self):
        expected = _expected_pins()
        if expected is None:
            self.skipTest("HF_IMAGE_CONFIG not set")
        self.pins, self.framework_version = expected

    def test_hf_stack_matches_config(self):
        for package, version in self.pins.items():
            with self.subTest(package=package):
                self.assertEqual(importlib.metadata.version(package), version)

    def test_torch_matches_framework_version(self):
        # Local segment (e.g. +cu130) is the wheel variant, not the release.
        torch_release = importlib.metadata.version("torch").split("+")[0]
        self.assertEqual(torch_release, self.framework_version)


class TestDependencyConsistency(unittest.TestCase):
    def test_no_broken_requirements(self):
        """The HF layer must not leave the environment with unsatisfied requirements."""
        uv = shutil.which("uv")
        if uv is None:
            self.skipTest("uv not installed")
        result = subprocess.run(
            [uv, "pip", "check", "--python", sys.executable],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class TestImports(unittest.TestCase):
    def test_stack_imports(self):
        for module in (
            "torch",
            "huggingface_hub",
            "transformers",
            "trl",
            "peft",
            "datasets",
            "accelerate",
        ):
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_trl_sft_imports(self):
        from trl import SFTConfig, SFTTrainer  # noqa: F401


@unittest.skipUnless(DEVICE_TYPE == "gpu", "requires a GPU image config")
class TestGpuImports(unittest.TestCase):
    def test_stack_imports(self):
        for module in (
            "evaluate",
            "diffusers",
            "math_verify",
            "deepspeed",
            "liger_kernel",
            "bitsandbytes",
        ):
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_trl_trainers_import(self):
        from trl import (  # noqa: F401
            DPOConfig,
            DPOTrainer,
            GRPOConfig,
            GRPOTrainer,
            RewardTrainer,
            SFTConfig,
            SFTTrainer,
        )

    def test_trl_supports_installed_vllm(self):
        """trl must accept the pinned vllm (it warns, not fails, when out of range)."""
        from trl.import_utils import is_vllm_available

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertTrue(is_vllm_available())
        unsupported = [w for w in caught if "supports vLLM versions" in str(w.message)]
        self.assertEqual(unsupported, [], [str(w.message) for w in unsupported])

    def test_trl_vllm_integration_imports(self):
        """trl's vllm modules import vllm symbols at module level; API drift fails here."""
        for module in (
            "trl.generation.vllm_client",
            "trl.generation.vllm_generation",
            "trl.experimental.async_grpo.weight_transfer",
        ):
            with self.subTest(module=module):
                importlib.import_module(module)


@unittest.skipUnless(DEVICE_TYPE == "neuronx", "requires a NeuronX image config")
class TestNeuronImports(unittest.TestCase):
    def test_neuron_import(self):
        importlib.import_module("torch_neuronx")


class TestCpuTraining(unittest.TestCase):
    """CPU LoRA training and TRL SFT construction with a local tiny model."""

    @classmethod
    def setUpClass(cls):
        cls.workdir = tempfile.mkdtemp(prefix="hf-sanity-")
        cls.addClassCleanup(shutil.rmtree, cls.workdir)
        cls.model_path = os.path.join(cls.workdir, "tiny-qwen2")
        cls.tokenizer = _build_tiny_model(cls.model_path)

    def _common_args(self, name):
        return dict(
            output_dir=os.path.join(self.workdir, name),
            max_steps=2,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_cpu=True,
            bf16=False,
        )

    @staticmethod
    def _lora_config():
        from peft import LoraConfig

        return LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
        )

    def test_transformers_trainer_lora_updates_adapters(self):
        import torch
        from datasets import Dataset
        from peft import get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        tokenizer = self.tokenizer
        model = get_peft_model(
            AutoModelForCausalLM.from_pretrained(self.model_path), self._lora_config()
        )
        dataset = Dataset.from_dict(
            {"text": ["the quick brown fox jumps over the lazy dog"] * 8}
        ).map(lambda batch: tokenizer(batch["text"]), batched=True, remove_columns=["text"])
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**self._common_args("trainer")),
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        lora_b_before = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if "lora_B" in name
        }
        self.assertTrue(lora_b_before, "no LoRA adapters were injected")
        trainer.train()

        losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
        self.assertTrue(losses, "trainer logged no loss")
        self.assertTrue(all(math.isfinite(loss) for loss in losses), losses)
        changed = [
            name
            for name, param in model.named_parameters()
            if name in lora_b_before and not torch.equal(param, lora_b_before[name])
        ]
        self.assertTrue(changed, "LoRA weights did not update")

    def test_trl_sft_trainer_builds_with_lora(self):
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        trainer = SFTTrainer(
            model=self.model_path,
            args=SFTConfig(**self._common_args("sft"), max_length=32),
            processing_class=self.tokenizer,
            train_dataset=Dataset.from_dict(
                {"text": ["the quick brown fox jumps over the lazy dog"] * 8}
            ),
            peft_config=self._lora_config(),
        )
        self.assertIn("input_ids", trainer.train_dataset.column_names)
        self.assertTrue(any("lora_B" in name for name, _ in trainer.model.named_parameters()))

    @unittest.skipUnless(DEVICE_TYPE == "gpu", "requires a GPU image config")
    def test_trl_grpo_trainer_builds(self):
        """Construction loads the model and prepares the reward and generation setup."""
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer

        def reward_len(completions, **kwargs):
            return [-abs(4 - len(c.split())) for c in completions]

        GRPOTrainer(
            model=self.model_path,
            reward_funcs=reward_len,
            args=GRPOConfig(
                **self._common_args("grpo"), num_generations=2, max_completion_length=8
            ),
            processing_class=self.tokenizer,
            train_dataset=Dataset.from_dict({"prompt": ["what is two plus two"] * 8}),
        )


@unittest.skipUnless(DEVICE_TYPE == "gpu", "requires a GPU image config")
class TestEntrypoint(unittest.TestCase):
    def test_entrypoint_runs_user_command_with_hf_defaults(self):
        """Non-`train` args must run through the launch chain with hf_optimizations applied."""
        if not os.path.isfile(ENTRYPOINT):
            self.skipTest("not a Hugging Face PyTorch training image")
        env = {k: v for k, v in os.environ.items() if k != "TOKENIZERS_PARALLELISM"}
        # start_with_right_hostname.sh runs non-`train` args via `eval "$@"`, which
        # re-splits them; keep the command free of shell metacharacters.
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as probe:
            probe.write(
                "import os, trl\n"
                "print('HF-SANITY', os.environ['TOKENIZERS_PARALLELISM'], trl.__version__)\n"
            )
        try:
            result = subprocess.run(
                [ENTRYPOINT, "python3", probe.name],
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )
        finally:
            os.unlink(probe.name)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        expected = f"HF-SANITY false {importlib.metadata.version('trl')}"
        self.assertIn(expected, result.stdout, output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
