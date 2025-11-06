import sys
import warnings
import os
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    load,
)

setup(
    name="apex",
    version="0.1",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "tests",
            "examples",
            "apex.egg-info",
        )
    ),
    install_requires=[
        "packaging>20.6",
    ],
    description="PyTorch Extensions written by NVIDIA",
)
