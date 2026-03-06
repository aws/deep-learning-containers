import logging
import sys

from test_utils.logger import ColoredFormatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter())
console_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
