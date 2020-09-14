import subprocess
import sys
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))

CUDA_VERSIONS = '10.1'


def main():
    openmpi_path_for_efa = '/opt/amazon/openmpi'
    check_ring_single_node(openmpi_path_for_efa)
    return 0


def check_ring_single_node(openmpi_path_for_efa):
    LOGGER.info("Launching ring single node test")
    try:        
        command = "{0} -n 3 --host localhost --oversubscribe " \
                    "-x LD_LIBRARY_PATH=/usr/local/cuda-{1}/efa/lib:/usr/local/cuda-{1}/lib:" \
                    "/usr/local/cuda-{1}/lib64:/usr/local/cuda-{1}:$LD_LIBRARY_PATH -x RDMAV_FORK_SAFE=1 " \
                    "~/src/bin/efa-tests/efa-cuda-{1}/ring".format(openmpi_path_for_efa, version)
        LOGGER.info(command)
        subprocess.check_call(command, shell=True, executable="/bin/bash")
        LOGGER.info("Single node ring test successful")
    except subprocess.CalledProcessError:
        LOGGER.error("Error: Single node ring test failed.")
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass