from packaging.specifiers import SpecifierSet
from packaging.version import Version


# TensorFlow
# Throughput, unit: images/second
TENSORFLOW_TRAINING_CPU_SYNTHETIC_THRESHOLD = {"<2.0": 50, ">=2.0": 50}
TENSORFLOW_TRAINING_GPU_SYNTHETIC_THRESHOLD = {"<2.0": 5000, ">=2.0": 7000}
TENSORFLOW_TRAINING_GPU_IMAGENET_THRESHOLD = {"<2.0": 5000, ">=2.0": 7000}

# TensorFlow Resnet
# Throughput, unit: images/second
TENSORFLOW_TRAINING_RN50_HPU_SYNTHETIC_THRESHOLD = {">=2.0": 1520}

# TensorFlow BERT
# Throughput, unit: sentences/second
TENSORFLOW_TRAINING_BERT_HPU_THRESHOLD = {">=2.0": 40}

# TensorFlow MASKRCNN
# Throughput, unit: samples/second
TENSORFLOW_TRAINING_MASKRCNN_HPU_THRESHOLD = {">=2.0": 11}

# p99 latency, unit: second
TENSORFLOW_INFERENCE_CPU_THRESHOLD = {
    "<2.0": {
        "INCEPTION": 0.06, "RCNN-Resnet101-kitti": 0.65, "Resnet50v2": 0.35, "MNIST": 0.00045, "SSDResnet50Coco": 0.4,
    },
    ">=2.0,<2.4": {
        "INCEPTION": 0.06, "RCNN-Resnet101-kitti": 0.65, "Resnet50v2": 0.35, "MNIST": 0.00045, "SSDResnet50Coco": 0.4,
    },
    # Updated thresholds for TF 2.4.1 CPU from Vanilla TF 2.4
    ">=2.4": {
        "INCEPTION": 0.11, "RCNN-Resnet101-kitti": 2.1, "Resnet50v2": 0.35, "MNIST": 0.001, "SSDResnet50Coco": 1.2,
    },
}
TENSORFLOW_INFERENCE_GPU_THRESHOLD = {
    "<2.0": {
        "INCEPTION": 0.04, "RCNN-Resnet101-kitti": 0.06, "Resnet50v2": 0.014, "MNIST": 0.0024, "SSDResnet50Coco": 0.1,
    },
    ">=2.0": {
        "INCEPTION": 0.04, "RCNN-Resnet101-kitti": 0.06, "Resnet50v2": 0.014, "MNIST": 0.0024, "SSDResnet50Coco": 0.1,
    },
}

# Throughput, unit: images/second
TENSORFLOW_SM_TRAINING_CPU_1NODE_THRESHOLD = {">=2.0": 30}
TENSORFLOW_SM_TRAINING_CPU_4NODE_THRESHOLD = {">=2.0": 20}
TENSORFLOW_SM_TRAINING_GPU_1NODE_THRESHOLD = {">=2.0": 2500}
TENSORFLOW_SM_TRAINING_GPU_4NODE_THRESHOLD = {">=2.0": 2500}

# MXNet
# Throughput, unit: images/second
MXNET_TRAINING_CPU_CIFAR_THRESHOLD = {">=1.0": 1000}
MXNET_TRAINING_GPU_IMAGENET_THRESHOLD = {">=1.0": 4500}
MXNET_INFERENCE_CPU_IMAGENET_THRESHOLD = {">=1.0": 100}
MXNET_INFERENCE_GPU_IMAGENET_THRESHOLD = {">=1.0": 4500}

# Accuracy, unit: NA
MXNET_TRAINING_GPU_IMAGENET_ACCURACY_THRESHOLD = {">=1.0": 0.9}

# Latency, unit: sec/epoch
MXNET_TRAINING_GPU_IMAGENET_LATENCY_THRESHOLD = {">=1.0": 120}

# PyTorch
# Throughput, unit: images/second
PYTORCH_TRAINING_GPU_SYNTHETIC_THRESHOLD = {">=1.0": 2400}

# PyTorch RN50
# Throughput, unit: images/second
PYTORCH_TRAINING_RN50_HPU_SYNTHETIC_1_CARD_THRESHOLD = {">=1.0": 1590}
PYTORCH_TRAINING_RN50_HPU_SYNTHETIC_8_CARD_THRESHOLD = {">=1.0": 4670}

# PyTorch BERT
# Throughput, unit: sentences/second
PYTORCH_TRAINING_BERT_HPU_THRESHOLD = {">=1.0": 40}

# Training Time Cost, unit: second/epoch
PYTORCH_TRAINING_GPU_IMAGENET_THRESHOLD = {">=1.0": 660}

# p99 latency, unit: millisecond
PYTORCH_INFERENCE_CPU_THRESHOLD = {
    ">=1.0": {
        "ResNet18": 0.08,
        "VGG13": 0.45,
        "MobileNetV2": 0.06,
        "GoogleNet": 0.12,
        "DenseNet121": 0.15,
        "InceptionV3": 0.25,
    }
}
PYTORCH_INFERENCE_GPU_THRESHOLD = {
    ">=1.0": {
        "ResNet18": 0.0075,
        "VGG13": 0.004,
        "MobileNetV2": 0.013,
        "GoogleNet": 0.018,
        "DenseNet121": 0.04,
        "InceptionV3": 0.03,
    }
}


def get_threshold_for_image(framework_version, lookup_table):
    """
    Find the correct threshold value(s) for a given framework version and a dict from which to lookup values.

    :param framework_version: Framework version of the image being tested
    :param lookup_table: The relevant dict from one of the dicts defined in this script
    :return: Threshold value as defined by one of the dicts in this script
    """
    for spec, threshold_val in lookup_table.items():
        if Version(framework_version) in SpecifierSet(spec):
            return threshold_val
    raise KeyError(
        f"{framework_version} does not satisfy any version constraint available in "
        f"{lookup_table.keys()}"
    )
