import os
import re
import pytest

from test.test_utils import CONTAINER_TESTS_PREFIX, get_framework_and_version_from_tag, UBUNTU_18_HPU_DLAMI_US_WEST_2
from test.test_utils.ec2 import execute_ec2_training_performance_test, execute_ec2_habana_training_performance_test
from src.benchmark_metrics import (
    get_threshold_for_image,
    TENSORFLOW_TRAINING_CPU_SYNTHETIC_THRESHOLD,
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

@pytest.mark.skip(reason="[REmove before merging] For this PR")
@pytest.mark.integration("synthetic dataset")
@pytest.mark.model("resnet50")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_HPU_INSTANCE_TYPE], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_HPU_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize('cards_num', [1, 8])
def test_performance_tensorflow_rn50_hpu_synthetic(tensorflow_training_habana, ec2_connection, upload_habana_test_artifact, cards_num):
    execute_ec2_habana_training_performance_test(
        ec2_connection,
        tensorflow_training_habana,
        TF_PERFORMANCE_RN50_TRAINING_HPU_SYNTHETIC_CMD,
        data_source="synthetic",
        cards_num=cards_num,
    )

@pytest.mark.integration("squad dataset")
@pytest.mark.model("bert")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_HPU_INSTANCE_TYPE], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_HPU_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize('cards_num', [1, 8])
def test_performance_tensorflow_bert_hpu(tensorflow_training_habana, ec2_connection, upload_habana_test_artifact, cards_num):
    execute_ec2_habana_training_performance_test(
        ec2_connection,
        tensorflow_training_habana,
        TF_PERFORMANCE_BERT_TRAINING_HPU_CMD,
        data_source="squad",
        cards_num=cards_num,
    )

@pytest.mark.skip(reason="[REmove before merging] For this PR")
@pytest.mark.integration("coco_like dataset")
@pytest.mark.model("maskrcnn")
@pytest.mark.parametrize("ec2_instance_type", [TF_EC2_HPU_INSTANCE_TYPE], indirect=True)
@pytest.mark.parametrize("ec2_instance_ami", [UBUNTU_18_HPU_DLAMI_US_WEST_2], indirect=True)
@pytest.mark.parametrize('cards_num', [1])
def test_performance_tensorflow_maskrcnn_hpu(tensorflow_training_habana, ec2_connection, upload_habana_test_artifact, cards_num):
    execute_ec2_habana_training_performance_test(
        ec2_connection,
        tensorflow_training_habana,
        TF_PERFORMANCE_MASKRCNN_TRAINING_HPU_CMD,
        data_source="coco_like",
        cards_num=cards_num,
    )


def post_process_tensorflow_training_performance(connection, log_location):
    last_lines = connection.run(f"tail {log_location}").stdout.split("\n")
    throughput = 0
    for line in reversed(last_lines):
        if "images/sec:" in line:
            throughput = float(re.search(r"(images/sec:[ ]*)(?P<throughput>[0-9]+\.?[0-9]+)", line).group("throughput"))
            break
    if throughput == 0:
        throughput = threshold_avg_calculated_from_all_steps(connection, log_location)
    return {"Throughput": throughput}


def threshold_avg_calculated_from_all_steps(connection, log_location):
    """
    This is a temporary fix for the flaky benchmark tests. This method is only called when the
    benchmark tests do not sleep peacefully and hence get a throughput value printed as 0. This 
    method reads the throughput at each step and finds the average of all the throughput values.
    """
    lines = connection.run(f"cat {log_location}", hide=True).stdout.split("\n")
    lines_with_images_sec = 0
    throughput_sum = 0.0
    step_count_of_last_line = 0
    for line in lines:
        splitted_arr = line.split()
        if len(splitted_arr) >= 3 and splitted_arr[1] == "images/sec:":
            lines_with_images_sec += 1
            throughput_sum += float(splitted_arr[2])
            step_count_of_last_line = int(splitted_arr[0])

    throughput_average = throughput_sum / float(lines_with_images_sec)
    assert lines_with_images_sec == int(step_count_of_last_line / 10) + 1, "Number of steps not as expected!!"
    return throughput_average
