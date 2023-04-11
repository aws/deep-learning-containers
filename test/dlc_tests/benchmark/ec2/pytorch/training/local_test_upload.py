from test.test_utils.ec2 import (
    read_metric,
    put_metric_data,
)

import os
import time
import pytest
import subprocess
from packaging.version import Version

def read_upload_benchmarking_result_to_cw(metric_names, pth, instance_type="p4d.24xlarge", model_suite="huggingface", precision="amp", namespace="PyTorch/EC2/Benchmarks/TorchDynamo/Inductor"):
    
    dimensions = [
             {"Name": "InstanceType", "Value": instance_type},
             {"Name": "ModelSuite", "Value": model_suite},
             {"Name": "Precision", "Value": precision},
             {"Name": "WorkLoad", "Value": "Training"},
         ]
    for name in metric_names:
        if name == "speedup":
            value = read_metric(os.path.join(pth, "geomean.csv"))
            unit = "None"
        if name == "comp_time":
            value = read_metric(os.path.join(pth, "comp_time.csv"))
            unit = "Seconds"
        if name == "memory":
            value = read_metric(os.path.join(pth, "memory.csv"))
            unit = "None"
        if name == "pass_rate":
            value = read_metric(os.path.join(pth, "pass_rate.csv"))
            unit = "Percent"

        put_metric_data(name, namespace, unit, value, dimensions)

read_upload_benchmarking_result_to_cw(["speedup"], "~/torchbench", model_suite="torchbench")