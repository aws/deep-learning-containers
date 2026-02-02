# Note: for vanilla PyTorch inference. Not with EIA

import argparse
import numpy as np
import time
import torch
import torchvision
from transformers import BertModel, RobertaModel, DistilBertModel, Wav2Vec2Model, AutoModel


cpu_models = [
    "ResNet18",
    "MobileNet_V2",
    "GoogLeNet",
    "DenseNet121",
    "Inception_V3",
    "Bert_128",
    "Roberta_128",
    "DistilBert_128",
    "All-MPNet_128",
    "ASR",
]

gpu_models = [
    "VGG13",
    "MobileNet_V2",
    "GoogLeNet",
    "DenseNet121",
    "Inception_V3",
    "ResNet18",
    "ResNet50",
    "ViT_B_16",
    "Bert_128",
    "Bert_256",
    "Roberta_128",
    "Roberta_256",
    "DistilBert_128",
    "DistilBert_256",
    "All-MPNet_128",
    "All-MPNet_256",
    "ASR",
]


def get_model_and_input(model_name):
    if model_name in ["Bert_128", "Bert_256"]:
        # input (0, vocab_size, (batch_size, seq_length))
        model = BertModel.from_pretrained("bert-base-uncased")
        input_tensor = torch.randint(0, 30522, (1, 256 if "256" in model_name else 128))
    elif model_name in ["Roberta_128", "Roberta_256"]:
        # input (0, vocab_size, (batch_size, seq_length))
        model = RobertaModel.from_pretrained("roberta-base")
        input_tensor = torch.randint(0, 50265, (1, 256 if "256" in model_name else 128))
    elif model_name in ["DistilBert_128", "DistilBert_256"]:
        # input (0, vocab_size, (batch_size, seq_length))
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        input_tensor = torch.randint(0, 30522, (1, 256 if "256" in model_name else 128))
    elif model_name in ["All-MPNet_128", "All-MPNet_256"]:
        # input (0, vocab_size, (batch_size, seq_length))
        model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        input_tensor = torch.randint(0, 30522, (1, 256 if "256" in model_name else 128))
    elif model_name == "ASR":
        # Shape: (batch_size, num_samples), num_samples = sample_rate(16kHz audio) * duration(1 seconds)
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        input_tensor = torch.rand(1, 16000)
    else:
        # For vision models: input shape is (batch_size, channels, height, width)
        model_class = getattr(torchvision.models, model_name.lower())
        model = model_class(weights=f"{model_name}_Weights.DEFAULT")
        input_tensor = torch.rand(
            1,
            3,
            299 if model_name == "Inception_V3" else 224,
            299 if model_name == "Inception_V3" else 224,
        )
    return model, input_tensor


def run_inference(model_name, iterations, is_gpu, instance):
    device = torch.device("cuda" if is_gpu else "cpu")
    model, input_tensor = get_model_and_input(model_name)

    model.eval().to(device)
    input_tensor = input_tensor.to(device)

    inference_times = []

    with torch.no_grad():
        if is_gpu:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            for _ in range(iterations):
                starter.record()
                model(input_tensor)
                ender.record()
                torch.cuda.synchronize()
                inference_times.append(starter.elapsed_time(ender))
        else:
            for _ in range(iterations):
                start = time.perf_counter()
                model(input_tensor)
                end = time.perf_counter()
                inference_times.append((end - start) * 1000)  # Convert to ms

    for percentile in [50, 90, 99]:
        print(
            f"{instance}: {model_name}: p{percentile} Latency: {np.percentile(inference_times, percentile):.6f} msec"
        )


def main(args):
    if args.gpu:
        with torch.cuda.device(0):
            run_inference(args.model, args.iterations, args.gpu, args.instance)
    else:
        run_inference(args.model, args.iterations, args.gpu, args.instance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", "-i", type=int, required=True, help="Number of iterations")
    parser.add_argument("--model", "-m", type=str, required=False, help="Model to run")
    parser.add_argument("--instance", type=str, required=True, help="Instance type")
    parser.add_argument("--gpu", "-gpu", action="store_true", help="Use GPU")
    args = parser.parse_args()

    models = gpu_models if args.gpu else cpu_models

    if args.model:
        assert (
            args.model in models
        ), f"Model {args.model} not in {'GPU' if args.gpu else 'CPU'} models list"
        main(args)
    else:
        for model in models:
            args.model = model
            main(args)
