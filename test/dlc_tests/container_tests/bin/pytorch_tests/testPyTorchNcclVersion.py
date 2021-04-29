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
        tail = subprocess.Popen(['head', '-n1'], stdin=find.stdout, stdout=PIPE)
        result = subprocess.Popen(['sed', '-r', 's/^.*\.so\.//'], stdin=tail.stdout, stdout=PIPE)
        output, error = result.communicate()
        numbers = [int(x, 10) for x in output.decode("utf-8").strip().split('.')]
        assert len(numbers) == 3, \
            "Error: failed to parse system nccl version"
        system_nccl = numbers[0] * 1000 + numbers[1] * 100 + numbers[2]
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