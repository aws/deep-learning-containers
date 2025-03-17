# Note: for vanilla PyTorch inference. Not with EIA

import argparse
import numpy as np
import time
import torch
import torchvision
from transformers import BertModel, RobertaModel, DistilBertModel, Wav2Vec2Model, AutoModel

# Dictionary of list with model definition and sample input tensor
# Simulate input (do not use tokenlizer) and run one forward pass for each iteration
models = {
    "ResNet18": [torchvision.models.resnet18, torch.rand(1, 3, 224, 224)],
    "VGG13": [torchvision.models.vgg13, torch.rand(1, 3, 224, 224)],
    "MobileNet_V2": [torchvision.models.mobilenet_v2, torch.rand(1, 3, 224, 224)],
    "GoogLeNet": [torchvision.models.googlenet, torch.rand(1, 3, 224, 224)],
    "DenseNet121": [torchvision.models.densenet121, torch.rand(1, 3, 224, 224)],
    "Inception_V3": [torchvision.models.inception_v3, torch.rand(1, 3, 299, 299)],
    "ResNet50": [torchvision.models.resnet50, torch.rand(1, 3, 224, 224)],
    "ViT_B_16": [torchvision.models.vision_transformer.vit_b_16, torch.rand(1, 3, 224, 224)],
    # input (0, vocab_size, (batch_size, seq_length))
    "Bert_128": [BertModel.from_pretrained("bert-base-uncased"), torch.randint(0, 30522, (1, 128))],
    "Bert_256": [BertModel.from_pretrained("bert-base-uncased"), torch.randint(0, 30522, (1, 256))],
    "Roberta_128": [
        RobertaModel.from_pretrained("roberta-base"),
        torch.randint(0, 50265, (1, 128)),
    ],
    "Roberta_256": [
        RobertaModel.from_pretrained("roberta-base"),
        torch.randint(0, 50265, (1, 256)),
    ],
    "DistilBert_128": [
        DistilBertModel.from_pretrained("distilbert-base-uncased"),
        torch.randint(0, 30522, (1, 128)),
    ],
    "DistilBert_256": [
        DistilBertModel.from_pretrained("distilbert-base-uncased"),
        torch.randint(0, 30522, (1, 256)),
    ],
    "All-MPNet_128": [
        AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        torch.randint(0, 30522, (1, 128)),
    ],
    "All-MPNet_256": [
        AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2"),
        torch.randint(0, 30522, (1, 256)),
    ],
    #  Shape: (batch_size, num_samples), num_samples = sample_rate(16kHz audio)* duration(1 seconds)
    "ASR": [Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h"), torch.rand(1, 16000)],
}


def get_device(is_gpu):
    if is_gpu:
        print("Using GPU:")
        return torch.device("cuda")
    print("Using CPU:")
    return torch.device("cpu")


def run_inference(model_name, iterations, is_gpu, instance):
    device = get_device(is_gpu)
    if model_name in [
        "Bert_128",
        "Roberta_128",
        "DistilBert_128",
        "Bert_256",
        "Roberta_256",
        "DistilBert_256",
        "ASR",
        "All-MPNet_128",
        "All-MPNet_256",
    ]:
        model, input_tensor = models[model_name]
    else:
        model_class, input_tensor = models[model_name]
        pretrained_weights = f"{model_name}_Weights.DEFAULT"
        model = model_class(weights=pretrained_weights)

    model.eval()
    model = model.to(device)  # send model to target hardware
    input_tensor = input_tensor.to(device)  # send input tensor to target hardware
    inference_times = []

    with torch.no_grad():
        for i in range(iterations):
            start = time.time()
            model(input_tensor)
            end = time.time()
            inference_times.append(end - start)

    latency_mean = 0.0

    for percentile in [50, 90, 99]:
        print(
            "{}: {}: p{} Latency: {} msec".format(
                instance, model_name, percentile, np.percentile(inference_times, percentile)
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations",
        "-i",
        help="How many iterations to run inference for",
        type=int,
        required=True,
    )
    parser.add_argument("--model", "-m", help="Which model to run", type=str, required=False)
    parser.add_argument("--instance", help="Which instance to run", type=str, required=True)
    parser.add_argument("--gpu", "-gpu", help="Toggle if running on GPU", action="store_true")
    args = vars(parser.parse_args())
    iterations = args["iterations"]
    model_name = args["model"]
    is_gpu = args["gpu"]
    instance = args["instance"]

    if not model_name:
        for model_name in models.keys():
            run_inference(model_name, iterations, is_gpu, instance)
    else:
        assert model_name in models

        run_inference(model_name, iterations, is_gpu, instance)
