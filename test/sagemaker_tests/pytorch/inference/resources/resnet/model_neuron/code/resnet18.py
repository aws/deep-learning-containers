
import os
import torch
import torch.neuron
import io
import torchvision.transforms as transforms
from PIL import Image

def model_fn(model_dir):
    """Loads a model. Provides a default implementation.
    Users can provide customized model_fn() in script.
    Args:
        model_dir: a directory where model is saved.
    Returns: A PyTorch model.
    """

    model_files = []
    for f in os.listdir(model_dir):
        if os.path.isfile(f):
            name, ext = os.path.splitext(f)
            if ext == ".pt" or ext == ".pth":
                model_files.append(f)
    if len(model_files) != 1:
        raise ValueError("Exactly one .pth or .pt file is required for PyTorch models: {}".format(model_files))
    return torch.jit.load(model_files[0])


def input_fn(request_body, request_content_type):

    print("Type of request body is {} and content type is {}".format(type(request_body), request_content_type))
    f = io.BytesIO(request_body)
    print("nbytes is {}".format(f.getbuffer().nbytes))
    input_image = Image.open(f).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch
