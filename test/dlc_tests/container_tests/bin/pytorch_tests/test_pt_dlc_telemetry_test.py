import os
import numpy as np
import time
from packaging.version import Version


def _clean_up_reports():
    if os.path.exists("/tmp/test_request.txt"):
        os.system("rm /tmp/test_request.txt")
    if os.path.exists("/tmp/test_tag_request.txt"):
        os.system("rm /tmp/test_tag_request.txt")


def opt_in_opt_out_test(exec_cmd):
    os.environ["TEST_MODE"] = "1"

    for opt_out_value in ["True", "TRUE", "true"]:
        _clean_up_reports()
        os.environ["OPT_OUT_TRACKING"] = opt_out_value
        cmd = f"python -c '{exec_cmd}'"
        os.system(cmd)
        time.sleep(5)
        assert not os.path.exists(
            "/tmp/test_request.txt"
        ), f"URL request placed even though OPT_OUT_TRACKING is {opt_out_value}."
        assert not os.path.exists(
            "/tmp/test_tag_request.txt"
        ), f"Tag request placed even though OPT_OUT_TRACKING is {opt_out_value}."

    for opt_out_value in ["False", "XYgg"]:
        _clean_up_reports()
        os.environ["OPT_OUT_TRACKING"] = opt_out_value
        cmd = f"python -c '{exec_cmd}'"
        os.system(cmd)
        time.sleep(5)
        assert os.path.exists(
            "/tmp/test_request.txt"
        ), f"URL request not placed even though OPT_OUT_TRACKING is {opt_out_value}."
        assert os.path.exists(
            "/tmp/test_tag_request.txt"
        ), f"Tag request not placed even though OPT_OUT_TRACKING is {opt_out_value}."

    os.environ["TEST_MODE"] = "0"
    print("Opt-In/Opt-Out Test passed")


def perf_test(exec_cmd):
    os.environ["TEST_MODE"] = "0"
    os.environ["OPT_OUT_TRACKING"] = "False"
    NUM_ITERATIONS = 5

    for itr in range(NUM_ITERATIONS):
        total_time_in = 0
        for x in range(NUM_ITERATIONS):
            cmd = f"python -c '{exec_cmd}'"
            start = time.time()
            os.system(cmd)
            total_time_in += time.time() - start
        print("avg out time: ", total_time_in / NUM_ITERATIONS)

        total_time_out = 0
        for x in range(NUM_ITERATIONS):
            cmd = f"export OPT_OUT_TRACKING='true' && python -c '{exec_cmd}'"
            start = time.time()
            os.system(cmd)
            total_time_out += time.time() - start
        print("avg out time: ", total_time_out / NUM_ITERATIONS)

        np.testing.assert_allclose(
            total_time_in / NUM_ITERATIONS, total_time_out / NUM_ITERATIONS, rtol=0.2, atol=0.5
        )

        print("DLC Telemetry performance test Passed")


perf_test("import torch")
opt_in_opt_out_test("import torch")

try:
    import torch

    torch_version = torch.__version__
except ImportError:
    raise ImportError("PyTorch is not installed or cannot be imported.")

# TEMP: sitecustomize.py current exists in PyTorch 2.6 DLCs. Skip logic should be reverted once sitecustomize.py has been added to all DLCs
if Version(torch_version) >= Version("2.6"):
    print("PyTorch version is 2.6 or higher. Running OS tests...")
    perf_test("import os")
    opt_in_opt_out_test("import os")
    print("OS tests completed.")
else:
    print(
        "TEMP: sitecustomize.py current exists in PyTorch 2.6 DLCs. Skip logic should be reverted once sitecustomize.py has been added to all DLCs"
    )
    print("PyTorch version is below 2.6. Skipping OS tests.")

print("All DLC telemetry test passed")
