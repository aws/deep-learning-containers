import subprocess
import sys
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def main():
    check_ib_uverbs()
    check_fi_info()
    check_efa_installed_packages()
    run_fi_pingpong_test()

    return 0


def check_ib_uverbs():
    """
    Following function checks for the presence of ib_uverbs module
    ib_uverbs stands for InfiniBand User Verbs.
    It enables direct userspace access to IB hardware via "verbs"
    """
    try:
        command = "lsmod | grep ib_uverbs"
        subprocess.check_call(command, shell=True, executable="/bin/bash")
        LOGGER.info("ib_uverbs module is present in the modules - test succeeded")
    except subprocess.CalledProcessError:
        LOGGER.error("Error: test check_ib_uverbs failed.")
        raise
    return True


def check_fi_info():
    """
    fi_info is a utility to query for fabric interfaces
    fi_info -p flag: filters fabric interfaces by the provider implementation
    """
    try:
        command = "fi_info -p efa"
        response = subprocess.check_output(command, shell=True, executable="/bin/bash")
        LOGGER.info(f"Efa provider info is present in response: {response}")
    except subprocess.CalledProcessError:
        LOGGER.error("Error: test Check Efa provider failed.")
        raise
    return True


def check_efa_installed_packages():
    """
    Following function checks if efa_installed_packages are present in /opt/amazon directory
    """
    try:
        command = "cat /opt/amazon/efa_installed_packages"
        subprocess.check_call(command, shell=True, executable="/bin/bash")
        LOGGER.info("efa packages are successfully installed in the correct path")
    except subprocess.CalledProcessError:
        LOGGER.error("Error: test check EFA packages failed.")
        raise
    return True


def run_fi_pingpong_test():
    """
    efa-tests suite performs ping-pong test to verify EFA installation
    """
    try:
        command = "cd ~ && ./src/bin/efa-tests/efa_test.sh"
        subprocess.check_call(command, shell=True, executable="/bin/bash")
        LOGGER.info("fi ping test is successful")
    except subprocess.CalledProcessError:
        LOGGER.error("Error: fi ping test failed.")
        raise
    return True


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
