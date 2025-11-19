import time

import pytest


def test_basic_logging(logger):
    """Test basic logging functionality"""
    logger.info("Starting basic logging test")

    # Log different levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    assert True
    logger.info("Test completed successfully")


@pytest.mark.parametrize("input,expected", [(2, 4), (3, 9), (4, 16)])
def test_parametrized(logger, input, expected):
    """Parametrized test with logging"""
    logger.info(f"Testing square of {input}")
    result = input * input

    assert result == expected
    logger.info(f"Square test passed: {input}² = {expected}")


def test_error_handling(logger):
    """Test error handling and logging"""
    logger.info("Starting error handling test")

    try:
        raise ValueError("This is a test error")
    except ValueError as e:
        logger.error(f"Caught error: {str(e)}")
        logger.exception("Full traceback:")
