import time

import pytest
from rich import print as rprint
from rich.progress import track
from rich.table import Table


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


def test_with_rich_table(logger):
    """Test using rich table with logging"""
    logger.info("Starting table test")

    # Create and display a rich table
    table = Table(title="Test Results")
    table.add_column("Test Case", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Duration", style="blue")

    table.add_row("Login Test", "✅ Passed", "0.5s")
    table.add_row("API Test", "✅ Passed", "1.2s")

    rprint(table)
    logger.info("Table test completed")


def test_with_progress(logger):
    """Test with progress tracking"""
    logger.info("Starting progress test")

    # Simulate work with progress tracking
    results = []
    for i in track(range(5), description="Processing items..."):
        time.sleep(0.2)
        results.append(i)
        logger.debug(f"Processed item {i}")

    assert len(results) == 5
    logger.info("Progress test completed")


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
