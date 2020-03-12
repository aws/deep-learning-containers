import subprocess

import pytest
from retrying import retry


def run_subprocess_cmd(cmd, failure="Command failed"):
    command = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    if command.returncode:
        pytest.fail(f"{failure}. Error log:\n{command.stdout.decode()}")
    return command


def retry_if_result_is_false(result):
    """Return True if we should retry (in this case retry if the result is False), False otherwise"""
    return result is False


@retry(stop_max_attempt_number=10, wait_fixed=2000, retry_on_result=retry_if_result_is_false)
def request_mxnet_inference_squeezenet(ip_address="127.0.0.1", port="80"):
    run_subprocess_cmd("curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    run_out = run_subprocess_cmd(f"curl -X POST http://{ip_address}:{port}/predictions/squeezenet -T kitten.jpg")

    # The run_out.return_code is not reliable, since sometimes predict request may succeed but the returned result
    # is 404. Hence the extra check.
    if run_out.returncode != 0 or 'probability' not in run_out.stdout:
        return False

    return True


def get_mms_run_command(model_names, processor="cpu"):
    """Helper function to format run command for MMS

    Args:
        Required - model_names (must have an entry in mxnet_model_location dict): str or list of str
    Returns:
        Command to start MMS server with given model
    """
    mxnet_model_location = {
        "squeezenet": "https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model",
        "pytorch-densenet": "https://asimov-multi-model-server.s3.amazonaws.com/pytorch/densenet/densenet.mar",
        "bert_sst": "https://aws-dlc-sample-models.s3.amazonaws.com/bert_sst/bert_sst.mar"
    }

    if not isinstance(model_names, list):
        model_names = [model_names]

    for model_name in model_names:
        if model_name not in mxnet_model_location:
            raise Exception("No entry found for model {} in dictionary".format(model_name))

    parameters = ["{}={}".format(name, mxnet_model_location[name]) for name in model_names]
    mms_command = ("mxnet-model-server --start --mms-config /home/model-server/config.properties "
                   "--models " + " ".join(parameters))
    return mms_command
