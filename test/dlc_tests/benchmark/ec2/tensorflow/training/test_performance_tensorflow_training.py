import os
import re
import pytest
import statistics

from test.test_utils import CONTAINER_TESTS_PREFIX, get_framework_and_version_from_tag, HPU_AL2_DLAMI, get_synapseai_version_from_tag
from test.test_utils.ec2 import execute_ec2_training_performance_test
from src.benchmark_metrics import (
    get_threshold_for_image,
    TENSORFLOW_TRAINING_CPU_SYNTHETIC_THRESHOLD,
    TENSORFLOW_TRAINING_RN50_HPU_SYNTHETIC_THRESHOLD,
    TENSORFLOW_TRAINING_BERT_HPU_THRESHOLD,
    TENSORFLOW_TRAINING_MASKRCNN_HPU_THRESHOLD,
    TENSORFLOW_TRAINING_GPU_SYNTHETIC_THRESHOLD,
    TENSORFLOW_TRAINING_GPU_IMAGENET_THRESHOLD,
)

TF_PERFORMANCE_TRAINING_CPU_SYNTHETIC_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_training_performance_cpu"
)
TF_PERFORMANCE_RN50_TRAINING_HPU_SYNTHETIC_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_rn50_training_performance_hpu_synthetic"
)
TF_PERFORMANCE_BERT_TRAINING_HPU_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_bert_training_performance_hpu"
)
TF_PERFORMANCE_MASKRCNN_TRAINING_HPU_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_maskrcnn_training_performance_hpu"
)
TF_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_training_performance_gpu_synthetic",
)
TF_PERFORMANCE_TRAINING_GPU_IMAGENET_CMD = os.path.join(
    CONTAINER_TESTS_PREFIX, "benchmark", "run_tensorflow_training_performance_gpu_imagenet",
)

TF_EC2_GPU_INSTANCE_TYPE = "p3.16xlarge"
TF_EC2_CPU_INSTANCE_TYPE = "c5.18xlarge"
TF_EC2_HPU_INSTANCE_TYPE = "dl1.24xlarge"


@pytest.mark.integration("synthetic dataset")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_CPU_INSTANCE_TYPE], indirect=True)
def test_performance_tensorflow_cpu(tensorflow_training, ec2_connection, cpu_only):
    _, framework_version = get_framework_and_version_from_tag(tensorflow_training)
    threshold = get_threshold_for_image(framework_version, TENSORFLOW_TRAINING_CPU_SYNTHETIC_THRESHOLD)
    execute_ec2_training_performance_test(
        ec2_connection,
        tensorflow_training,
        TF_PERFORMANCE_TRAINING_CPU_SYNTHETIC_CMD,
        post_process=post_process_tensorflow_training_performance,
        data_source="synthetic",
        threshold={"Throughput": threshold},
    )


# TODO: Enable TF1 by removing "tf2_only" fixture once infrastructure issues have been resolved.
@pytest.mark.integration("synthetic dataset")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_performance_tensorflow_gpu_synthetic(tensorflow_training, ec2_connection, gpu_only, tf2_only):
    _, framework_version = get_framework_and_version_from_tag(tensorflow_training)
    threshold = get_threshold_for_image(framework_version, TENSORFLOW_TRAINING_GPU_SYNTHETIC_THRESHOLD)
    execute_ec2_training_performance_test(
        ec2_connection,
        tensorflow_training,
        TF_PERFORMANCE_TRAINING_GPU_SYNTHETIC_CMD,
        post_process=post_process_tensorflow_training_performance,
        data_source="synthetic",
        threshold={"Throughput": threshold},
    )


# TODO: Enable TF1 by removing "tf2_only" fixture once infrastructure issues have been resolved.
@pytest.mark.integration("imagenet dataset")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_GPU_INSTANCE_TYPE], indirect=True)
def test_performance_tensorflow_gpu_imagenet(tensorflow_training, ec2_connection, gpu_only, tf2_only):
    _, framework_version = get_framework_and_version_from_tag(tensorflow_training)
    threshold = get_threshold_for_image(framework_version, TENSORFLOW_TRAINING_GPU_IMAGENET_THRESHOLD)
    execute_ec2_training_performance_test(
        ec2_connection,
        tensorflow_training,
        TF_PERFORMANCE_TRAINING_GPU_IMAGENET_CMD,
        post_process=post_process_tensorflow_training_performance,
        data_source="imagenet",
        threshold={"Throughput": threshold},
    )

@pytest.mark.integration("synthetic dataset")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_HPU_INSTANCE_TYPE], indirect=True)
# TODO: Ensure 8 card instance is used
@pytest.mark.parametrize("ec2_instance_ami", [HPU_AL2_DLAMI], indirect=True)
@pytest.mark.parametrize('cards_num', [1, 8])
def test_performance_tensorflow_rn50_hpu_synthetic(tensorflow_training_habana, ec2_connection, cards_num):
    _, framework_version = get_framework_and_version_from_tag(tensorflow_training_habana)
    synapseai_version = get_synapseai_version_from_tag(tensorflow_training_habana)
    threshold = get_threshold_for_image(framework_version, TENSORFLOW_TRAINING_RN50_HPU_SYNTHETIC_THRESHOLD)
    perf_factor = 0.95 if cards_num > 1 else 1  # Rough scaling factor
    expected_perf = threshold * cards_num * perf_factor
    allowed_regression = expected_perf * 0.1
    threshold = expected_perf - allowed_regression
    execute_ec2_training_performance_test(
        ec2_connection,
        tensorflow_training_habana,
        TF_PERFORMANCE_RN50_TRAINING_HPU_SYNTHETIC_CMD,
        post_process=post_process_tensorflow_rn50_hpu_training_performance,
        data_source="synthetic",
        threshold={"Throughput": threshold},
        cards_num=cards_num,
        synapseai_version=synapseai_version
    )

@pytest.mark.integration("squad dataset")
@pytest.mark.model("bert")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_HPU_INSTANCE_TYPE], indirect=True)
# TODO: Ensure 8 card instance is used
@pytest.mark.parametrize("ec2_instance_ami", [HPU_AL2_DLAMI], indirect=True)
@pytest.mark.parametrize('cards_num', [1, 8])
def test_performance_tensorflow_bert_hpu(tensorflow_training_habana, ec2_connection, cards_num):
    _, framework_version = get_framework_and_version_from_tag(tensorflow_training_habana)
    synapseai_version = get_synapseai_version_from_tag(tensorflow_training_habana)
    threshold = get_threshold_for_image(framework_version, TENSORFLOW_TRAINING_BERT_HPU_THRESHOLD)
    perf_factor = 0.95 if cards_num > 1 else 1  # Rough scaling factor
    expected_perf = threshold * cards_num * perf_factor
    allowed_regression = expected_perf * 0.1
    threshold = expected_perf - allowed_regression
    execute_ec2_training_performance_test(
        ec2_connection,
        tensorflow_training_habana,
        TF_PERFORMANCE_BERT_TRAINING_HPU_CMD,
        post_process=post_process_tensorflow_bert_hpu_training_performance,
        data_source="squad",
        threshold={"Throughput": threshold},
        cards_num=cards_num,
        synapseai_version=synapseai_version
    )

@pytest.mark.integration("coco_like dataset")
@pytest.mark.model("maskrcnn")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_HPU_INSTANCE_TYPE], indirect=True)
# TODO: Ensure 8 card instance is used
@pytest.mark.parametrize("ec2_instance_ami", [HPU_AL2_DLAMI], indirect=True)
@pytest.mark.parametrize('cards_num', [1])
def test_performance_tensorflow_maskrcnn_hpu(tensorflow_training_habana, ec2_connection, cards_num):
    _, framework_version = get_framework_and_version_from_tag(tensorflow_training_habana)
    synapseai_version = get_synapseai_version_from_tag(tensorflow_training_habana)
    threshold = get_threshold_for_image(framework_version, TENSORFLOW_TRAINING_MASKRCNN_HPU_THRESHOLD)
    perf_factor = 0.95 if cards_num > 1 else 1  # Rough scaling factor
    expected_perf = threshold * cards_num * perf_factor
    allowed_regression = expected_perf * 0.1
    threshold = expected_perf - allowed_regression
    execute_ec2_training_performance_test(
        ec2_connection,
        tensorflow_training_habana,
        TF_PERFORMANCE_MASKRCNN_TRAINING_HPU_CMD,
        post_process=post_process_tensorflow_maskrcnn_hpu_training_performance,
        data_source="coco_like",
        threshold={"Throughput": threshold},
        cards_num=cards_num,
        synapseai_version=synapseai_version
    )

def post_process_tensorflow_rn50_hpu_training_performance(connection, log_location):
    lines = connection.run(f"cat {log_location}").stdout.split("\n")
    throughput = 0.0
    step_num = '0'
    eps = dict()
    iteration_num = 0
    for line in lines:
        if "examples/second between" in line:
                perf_line = re.compile(r"] .+ (\d+\.\d+) examples/second between steps (\d+) and (\d+)")
                res_line = perf_line.search(line)
                if res_line:
                    step_num = str(res_line.group(2))
                    # Skip first 8 prints due to compilation time
                    if iteration_num < 8:
                        iteration_num += 1
                    else:
                        if step_num not in eps.keys():
                            eps[step_num] = list()
                        eps[step_num].append(float(res_line.group(1)))
    avg_perf = [x[0] for x in eps.values()]
    throughput = round(statistics.mean(avg_perf), 2)
    return {"Throughput": throughput}

def post_process_tensorflow_bert_hpu_training_performance(connection, log_location):
    lines = connection.run(f"cat {log_location}").stdout.split("\n")
    throughput = 0.0
    train_perf = False
    perf_line = re.compile(r"Avg total throughput\s+=\s+(\S+)\s\[examples/sec\]")
    for line in lines:
        if 'train statistics' in line:
            train_perf = True
        obj = perf_line.search(line)
        if obj and train_perf:
            throughput = float((obj.group(1)))
            break
    return {"Throughput": throughput}

def post_process_tensorflow_maskrcnn_hpu_training_performance(connection, log_location):
    lines = connection.run(f"cat {log_location}").stdout.split("\n")
    throughput = 0.0
    train_perf = False
    perf_line = re.compile(r": Average throughput: (-?\d+\.\d+) samples/sec")
    for line in lines:
        if 'Training Performance Summary' in line:
            train_perf = True
        obj = perf_line.search(line)
        if obj and train_perf:
            throughput = float((obj.group(1)))
            break
    return {"Throughput": throughput}

def post_process_tensorflow_training_performance(connection, log_location):
    last_lines = connection.run(f"tail {log_location}").stdout.split("\n")
    throughput = 0
    for line in reversed(last_lines):
        if "images/sec:" in line:
            throughput = float(re.search(r"(images/sec:[ ]*)(?P<throughput>[0-9]+\.?[0-9]+)", line).group("throughput"))
            break
    return {"Throughput": throughput}
