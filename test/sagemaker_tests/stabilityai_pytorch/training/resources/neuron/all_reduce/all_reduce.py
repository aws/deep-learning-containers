import os
import subprocess
import torch_xla.core.xla_model as xm
import torch
import torch_xla.distributed.xla_backend

torch.distributed.init_process_group("xla")
import torch_xla.distributed.xla_multiprocessing as xmp
import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _mp_fn():
    os.environ["NEURON_CC_FLAGS"] = (
        os.environ.get("NEURON_CC_FLAGS", "") + "--cache_dir=neff_cache2"
    )
    os.environ["FI_EFA_USE_DEVICE_RDMA"] = "1"
    os.environ["FI_PROVIDER"] = "efa"
    os.environ["NCCL_DEBUG"] = "TRACE"
    os.environ["NCCL_INIT"] = "TRACE"
    os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    os.environ["NCCL_SOCKET_IFNAME"] = os.environ["SM_NETWORK_INTERFACE_NAME"]
    os.environ["NEURON_RT_LOG_LEVEL"] = "INFO"

    world_size = xm.xrt_world_size()
    device = xm.xla_device()
    rank = xm.get_ordinal()
    ones = torch.ones((2, 3))
    xones = ones.to(device)
    if world_size > 0:
        print("running all reduce")
        for i in range(0, 5):
            print(f"at iteration {i}, with local rank {rank}", flush=True)
            result = xm.all_reduce(xm.REDUCE_SUM, xones)
            result_cpu = result.cpu()
            xm.mark_step()
            print(result_cpu, flush=True)
        expected = torch.ones((2, 3)) * world_size
        assert expected.allclose(result_cpu)
        logger.info("PASS")


if __name__ == "__main__":
    _mp_fn()
