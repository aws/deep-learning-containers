import os
import time
import pytest
from test.test_utils import BENCHMARK_RESULTS_S3_BUCKET, is_tf1, is_tf20, framework_short_version


@pytest.mark.model("inception, RCNN-Resnet101-kitti, resnet50_v2, mnist, SSDResnet50Coco")
@pytest.mark.parametrize("ec2_instance_type", ["p3.16xlarge"], indirect=True)
def test_performance_ec2_tensorflow_inference_gpu(tensorflow_inference, ec2_connection, region, gpu_only):
    ec2_performance_tensorflow_inference(tensorflow_inference, "gpu", ec2_connection, region)


@pytest.mark.model("inception, RCNN-Resnet101-kitti, resnet50_v2, mnist, SSDResnet50Coco")
@pytest.mark.parametrize("ec2_instance_type", ["c5.18xlarge"], indirect=True)
def test_performance_ec2_tensorflow_inference_cpu(tensorflow_inference, ec2_connection, region, cpu_only):
    ec2_performance_tensorflow_inference(tensorflow_inference, "cpu", ec2_connection, region)


def ec2_performance_tensorflow_inference(image_uri, processor, ec2_connection, region):
    docker_cmd = "nvidia-docker" if processor == "gpu" else "docker"
    python_version = "py2" if "py2" in image_uri else "py3"
    container_test_local_dir = os.path.join("$HOME", "container_tests")
    tf_major_version = "1" if is_tf1(image_uri) else "2"
    tf_api_version = framework_short_version(image_uri)
    tf_version_folder = '1.15' if tf_major_version == '1' else '2.1'
    tf_script_version = tf_major_version if not is_tf20(image_uri) else "20"

    processor_folder = "CPU-WITH-MKL" if processor == "cpu" else "GPU"

    # Make sure we are logged into ECR so we can pull the image
    ec2_connection.run(f"$(aws ecr get-login --no-include-email --region {region})", hide=True)

    ec2_connection.run(f"{docker_cmd} pull -q {image_uri} ")
    ec2_connection.run(f"pip3 install -U pip")

    return_val = ec2_connection.run(
            f"pip3 install boto3 grpcio tensorflow-serving-api=={tf_api_version} --user --no-warn-script-location", warn=True
        )
    if return_val != 0:  # in case tfs version is behind tf version
        ec2_connection.run(f"echo tfs version is behind tf version  >&2")
        latest_tfs_api = '"tensorflow-serving-api<2"' if tf_major_version == "1" else '"tensorflow-serving-api>=2"'
        ec2_connection.run(
            f'pip3 install boto3 grpcio {latest_tfs_api} --user --no-warn-script-location'
        )
    ec2_connection.sudo(
        f"aws s3 cp s3://tensorflow-aws/{tf_version_folder}/Serving/{processor_folder}/tensorflow_model_server /usr/bin/")
    ec2_connection.sudo(f"chmod +x /usr/bin/tensorflow_model_server")
    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    commit_info = os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION")
    log_file = f"inference_benchmark_results_{commit_info}_{time_str}.log"
    ec2_connection.run(
        f"python3 {container_test_local_dir}/bin/benchmark/tf{tf_script_version}_serving_perf.py "
        f"--processor {processor} --docker_image_name {image_uri} --run_all_s3 --binary /usr/bin/tensorflow_model_server --get_perf --iterations 1000 "
        f"2>&1 | tee {log_file}"
    )
    ec2_connection.run(
        f"echo Benchmark Results: >&2;"
        f"echo Tensorflow{tf_major_version} Inference {processor} {python_version} >&2"
    )
    ec2_connection.run(f"tail {log_file} >&2")
    ec2_connection.run(
        f"aws s3 cp {log_file} {BENCHMARK_RESULTS_S3_BUCKET}/tensorflow{tf_major_version}/ec2/inference/{processor}/{python_version}/{log_file}")
    ec2_connection.run(
        f"echo To retrieve complete benchmark log, check s3://dlinfra-dlc-cicd-performance/tensorflow{tf_major_version}/ec2/inference/{processor}/{python_version}/{log_file} >&2")
