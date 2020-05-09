# Note: for vanilla PyTorch inference. Not with EIA

import argparse
import numpy as np
import time
import torch
import torchvision

# Dictionary of list with model definition and sample input
# models = {'ResNet18': [torchvision.models.resnet18, torch.rand(1,3,224,224)],
#      'VGG13': [torchvision.models.vgg13, torch.rand(1,3,224,224)],
#      'MobileNetV2': [torchvision.models.mobilenet_v2, torch.rand(1,3,224,224)],
#      'GoogleNet': [torchvision.models.googlenet, torch.rand(1,3,224,224)],
#      'DenseNet121': [torchvision.models.densenet121, torch.rand(1,3,224,224)],
#      'InceptionV3': [torchvision.models.inception_v3, torch.rand(1,3,299,299)],
#      'DeepLabV3_ResNet50': [torchvision.models.segmentation.deeplabv3_resnet50, torch.rand(10,3,224,224)],
#      'FCN_ResNet50': [torchvision.models.segmentation.fcn_resnet50, torch.rand(10,3,224,224)]}
models = {'ResNet18': [torchvision.models.resnet18, torch.rand(1, 3, 224, 224)],
          'VGG13': [torchvision.models.vgg13, torch.rand(1, 3, 224, 224)],
          'MobileNetV2': [torchvision.models.mobilenet_v2, torch.rand(1, 3, 224, 224)],
          'GoogleNet': [torchvision.models.googlenet, torch.rand(1, 3, 224, 224)],
          'DenseNet121': [torchvision.models.densenet121, torch.rand(1, 3, 224, 224)],
          'InceptionV3': [torchvision.models.inception_v3, torch.rand(1, 3, 299, 299)]
          }


def get_device(is_gpu):
    if is_gpu:
        print("Using GPU:")
        return torch.device('cuda')
    print("Using CPU:")
    return torch.device('cpu')


def run_inference(model_name, iterations, is_gpu):
    model_class, input_tensor = models[model_name]
    device = get_device(is_gpu)
    model = model_class(pretrained=True)
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
        print('{}: p{} Latency: {} msec'.format(model_name, percentile, np.percentile(inference_times, percentile)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', '-i', help='How many iterations to run inference for', type=int, required=True)
    parser.add_argument('--model', '-m', help='Which model to run', type=str, required=False)
    parser.add_argument('--gpu', '-gpu', help='Toggle if running on GPU', action="store_true")
    args = vars(parser.parse_args())
    iterations = args['iterations']
    model_name = args['model']
    is_gpu = args['gpu']

    if not model_name:
        for model_name in models.keys():
            run_inference(model_name, iterations, is_gpu)

    assert (model_name in models)

    run_inference(model_name, iterations, is_gpu)

