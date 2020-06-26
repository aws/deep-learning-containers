import logging
import os
import sys

import pytest

from invoke.context import Context

from test.test_utils import is_pr_context, PR_ONLY_REASON

pytest.mark.skipif(not is_pr_context(), reason=PR_ONLY_REASON)
def test_binary_visibility():
    pass
