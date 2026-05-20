"""EC2 EFA integration test.

Launches 2x p4d.24xlarge with EFA, runs NCCL all_reduce_perf across nodes,
and verifies EFA transport is used (not sockets).

Ported from V1: test/dlc_tests/ec2/test_efa.py + test/v2/ec2/efa/test_efa.py

Usage:
    pytest test/efa/test_efa.py --image-uri <ecr-image-uri> -v
"""

import logging
import os

from efa.ec2_helpers import (
    DEFAULT_TIMEOUT,
    HOSTS_FILE_LOCATION,
    MASTER_CONTAINER_NAME,
    efa_instances,
    run_on_container,
)

LOGGER = logging.getLogger(__name__)

IMAGE_URI = os.environ["TEST_IMAGE_URI"]
EFA_INSTANCE_TYPE = os.environ.get("EFA_INSTANCE_TYPE", "p4d.24xlarge")


def test_efa_sanity_and_nccl(image_uri=IMAGE_URI):
    """Run EFA sanity checks and NCCL all_reduce_perf across 2 nodes.

    Verifies:
    - EFA provider detected (fi_info -p efa)
    - fi_pingpong over EFA loopback
    - RDMA devices present (ibv_devinfo)
    - GPU Direct RDMA (GDR) available
    - NCCL uses EFA transport ("Selected provider is efa")
    - NCCL uses Libfabric ("Using network Libfabric")
    - NCCL uses GDRDMA on p4d/p5 ("NET/Libfabric/0/GDRDMA")
    - all_reduce bandwidth >= 3 GB/s on the 1 GiB message size
    """
    with efa_instances(image_uri=image_uri, instance_type=EFA_INSTANCE_TYPE) as (
        master_conn,
        worker_conn,
        aws_session,
    ):
        # Diagnostics: dump NCCL plugin state before running the test
        diag = run_on_container(
            MASTER_CONTAINER_NAME,
            master_conn,
            "echo NCCL_NET_PLUGIN=$NCCL_NET_PLUGIN && "
            "ls -la /opt/amazon/ofi-nccl/lib64/ 2>&1 && "
            "ldd /opt/amazon/ofi-nccl/lib64/libnccl-net-ofi.so 2>&1 && "
            "echo --- && "
            "ls -la /opt/amazon/ofi-nccl/lib64/libnccl-net.so 2>&1 || true",
        )
        LOGGER.info("=== NCCL plugin diagnostics ===\n%s", diag.stdout)

        # EFA sanity on master
        run_on_container(
            MASTER_CONTAINER_NAME,
            master_conn,
            "/test/efa/scripts/efa_sanity.sh",
        )

        # NCCL all_reduce across 2 nodes
        run_on_container(
            MASTER_CONTAINER_NAME,
            master_conn,
            f"/test/efa/scripts/nccl_allreduce.sh {HOSTS_FILE_LOCATION} 2",
            timeout=DEFAULT_TIMEOUT,
        )
