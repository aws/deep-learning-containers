#!/usr/bin/env python
import sys
import subprocess
from subprocess import PIPE

import logging
import torch

LOGGER = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def test_nccl_version():
    try:
        pt_nccl = torch.cuda.nccl.version()
        find = subprocess.Popen(['find', '/usr', '-name', 'libnccl.so*'], stdout=PIPE)
        sort = subprocess.Popen(['sort', '-r'], stdin=find.stdout, stdout=PIPE)
        tail = subprocess.Popen(['head', '-n1'], stdin=sort.stdout, stdout=PIPE)
        result = subprocess.Popen(['sed', '-r', 's/^.*\.so\.//'], stdin=tail.stdout, stdout=PIPE)
        output, error = result.communicate()
        system_nccl = [int(x, 10) for x in output.decode("utf-8").strip().split('.')]
        assert len(system_nccl) == 3, \
            "Error: failed to parse system nccl version"
        if isinstance(pt_nccl, int):
            system_nccl = system_nccl[0] * 1000 + system_nccl[1] * 100 + system_nccl[2]
        elif isinstance(pt_nccl, tuple):
            pt_nccl = list(pt_nccl)
        else:
            raise AssertionError("Error: PT NCCL version should be either int or tuple")
        assert pt_nccl == system_nccl, \
            "Error: PT NCCL version does not match system NCCL: {} vs {}".format(pt_nccl, system_nccl)
    except Exception as excp:
        LOGGER.debug("Error: check_pytorch test failed.")
        LOGGER.debug("Exception: {}".format(excp))
        raise


if __name__ == '__main__':
    try:
        sys.exit(test_nccl_version())
    except KeyboardInterrupt:
        pass
