"""Run tiny Transformers, Sentence Transformers and Diffusers models on Neuron.

Every model is built from a config (no Hub access), runs on the `neuron` device,
and is checked against the same model on CPU.

Run on Trainium with:
    python transformers_neuron_inference.py --mode eager
    python transformers_neuron_inference.py --mode compiled

`eager` runs the operations one at a time on the device. `compiled` wraps the
model with `torch.compile(backend="neuron", dynamic=False)`.
"""

import argparse
import copy
import tempfile
from pathlib import Path

import torch

DEVICE = "neuron"
SEED = 42
# Neuron matmuls may accumulate differently than CPU, so compare loosely.
ATOL = 2e-2
RTOL = 2e-2


def to_neuron(module, mode):
    module = module.to(DEVICE).eval()
    if mode == "compiled":
        return torch.compile(module, backend="neuron", dynamic=False, fullgraph=True)
    return module


def assert_on_neuron(tensor, name):
    assert tensor.device.type == DEVICE, f"{name} ran on {tensor.device}, not on {DEVICE}"


def assert_close(actual, expected, name):
    assert_on_neuron(actual, name)
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    assert actual.shape == expected.shape, f"{name}: {actual.shape} != {expected.shape}"
    assert torch.isfinite(actual).all(), f"{name}: non-finite values on {DEVICE}"
    max_diff = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, atol=ATOL, rtol=RTOL), f"{name}: max diff {max_diff}"
    print(f"PASS {name} max_diff={max_diff:.2e}", flush=True)


def tiny_bert_config(**kwargs):
    from transformers import BertConfig

    return BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        **kwargs,
    )


def test_sequence_classification(mode):
    from transformers import BertForSequenceClassification

    torch.manual_seed(SEED)
    model = BertForSequenceClassification(tiny_bert_config(num_labels=3)).eval()
    input_ids = torch.randint(5, 128, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        expected = model(input_ids=input_ids, attention_mask=attention_mask).logits
        neuron_model = to_neuron(copy.deepcopy(model), mode)
        actual = neuron_model(
            input_ids=input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE)
        ).logits
    assert_close(actual, expected, f"{mode}/transformers/sequence-classification")


def test_causal_lm(mode):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(SEED)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3ForCausalLM(config).eval()
    input_ids = torch.randint(3, 128, (1, 8))
    with torch.inference_mode():
        expected = model(input_ids=input_ids, use_cache=False).logits
        neuron_model = to_neuron(copy.deepcopy(model), mode)
        actual = neuron_model(input_ids=input_ids.to(DEVICE), use_cache=False).logits
    assert_close(actual, expected, f"{mode}/transformers/causal-lm-forward")

    if mode == "eager":
        # Greedy decoding with a KV cache: the generated tokens must match CPU.
        generate_kwargs = dict(max_new_tokens=6, do_sample=False, min_new_tokens=6)
        with torch.inference_mode():
            expected_tokens = model.generate(input_ids, **generate_kwargs)
            actual_tokens = neuron_model.generate(input_ids.to(DEVICE), **generate_kwargs)
        assert_on_neuron(actual_tokens, "eager/transformers/generate")
        assert torch.equal(
            actual_tokens.cpu(), expected_tokens
        ), f"generate mismatch: {actual_tokens.cpu().tolist()} != {expected_tokens.tolist()}"
        print(f"PASS eager/transformers/generate tokens={actual_tokens.shape[-1]}", flush=True)


def save_tiny_bert(path, model_class_name="BertModel", **config_kwargs):
    """Save a tiny randomly initialized BERT model and a matching word-level tokenizer."""
    import transformers
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = "the a musical is best after hamilton considered to be of human history".split()
    vocab = {token: i for i, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])}
    vocab.update({word: i + len(vocab) for i, word in enumerate(words)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    torch.manual_seed(SEED)
    model_class = getattr(transformers, model_class_name)
    model_kwargs = {"add_pooling_layer": False} if model_class_name == "BertModel" else {}
    model_class(tiny_bert_config(**config_kwargs), **model_kwargs).save_pretrained(path)
    tokenizer.save_pretrained(path)


def test_sentence_transformers(mode):
    from sentence_transformers import SentenceTransformer, models

    sentences = [
        "hamilton is considered to be the best musical",
        "the best musical after hamilton is a history",
    ]
    with tempfile.TemporaryDirectory() as path:
        save_tiny_bert(path)

        def build(device):
            transformer = models.Transformer(path, max_seq_length=16)
            pooling = models.Pooling(transformer.get_word_embedding_dimension(), "mean")
            return SentenceTransformer(modules=[transformer, pooling], device=device).eval()

        # Fixed-length inputs keep the shapes static for torch.compile.
        encode_kwargs = dict(convert_to_tensor=True, batch_size=len(sentences))
        expected = build("cpu").encode(sentences, **encode_kwargs)
        model = build(DEVICE)
        model[0].tokenizer_args = {"padding": "max_length"}
        if mode == "compiled":
            model[0].auto_model = torch.compile(
                model[0].auto_model, backend="neuron", dynamic=False, fullgraph=True
            )
        actual = model.encode(sentences, **encode_kwargs)
    assert_close(actual, expected, f"{mode}/sentence-transformers/encode")


def test_diffusers_unet(mode):
    from diffusers import UNet2DConditionModel

    torch.manual_seed(SEED)
    unet = UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=8,
    ).eval()
    sample = torch.randn(1, 4, 16, 16)
    timestep = torch.tensor([10])
    encoder_hidden_states = torch.randn(1, 8, 32)
    with torch.inference_mode():
        expected = unet(sample, timestep, encoder_hidden_states).sample
        neuron_unet = to_neuron(copy.deepcopy(unet), mode)
        actual = neuron_unet(
            sample.to(DEVICE), timestep.to(DEVICE), encoder_hidden_states.to(DEVICE)
        ).sample
    assert_close(actual, expected, f"{mode}/diffusers/unet")


def main(mode):
    import torch_neuronx  # noqa: F401
    from transformers.utils import is_torch_neuron_available

    if not is_torch_neuron_available(check_device=True):
        raise RuntimeError("This inference test requires a working Neuron device")

    for test in (
        test_sequence_classification,
        test_causal_lm,
        test_sentence_transformers,
        test_diffusers_unet,
    ):
        test(mode)
    print(f"PASS all mode={mode}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    main(parser.parse_args().mode)
