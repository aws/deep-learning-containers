#!/usr/bin/env python3

import logging
import sys
from test.vllm_tests.infra.eks_infra import EksInfrastructure
from test.vllm_tests.test_artifacts.eks_test import VllmEksTest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_vllm_test():
    infrastructure = None
    try:
        logger.info("Setting up EKS infrastructure...")
        infrastructure = EksInfrastructure()
        if not infrastructure.setup_infrastructure():
            raise Exception("Infrastructure setup failed")
        logger.info("Infrastructure setup completed successfully")

        logger.info("Starting vLLM tests...")
        test = VllmEksTest()
        if not test.run_tests():
            raise Exception("vLLM tests failed")
        logger.info("vLLM tests completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1

    finally:
        if infrastructure:
            logger.info("Cleaning up infrastructure...")
            infrastructure.cleanup_infrastructure()
            logger.info("Cleanup completed")

def main():
    sys.exit(run_vllm_test())

if __name__ == "__main__":
    main()