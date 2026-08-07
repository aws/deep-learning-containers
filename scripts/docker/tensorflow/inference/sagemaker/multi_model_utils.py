# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import fcntl
import signal
import time
from contextlib import contextmanager

MODEL_CONFIG_FILE = "/sagemaker/model-config.cfg"
DEFAULT_LOCK_FILE = "/sagemaker/lock-file.lock"


@contextmanager
def lock(path=DEFAULT_LOCK_FILE, poll_interval=0.05, timeout=60.0):
    """File lock that yields to the gevent hub while contending.

    fcntl.lockf is NOT gevent-patched — the blocking form parks the entire
    hub, stalling every greenlet in the worker including /ping. Use the
    non-blocking form in a retry loop with time.sleep (which IS patched)
    so /ping keeps serving during model-load contention.
    """
    deadline = time.time() + timeout
    with open(path, "w", encoding="utf8") as f:
        fd = f.fileno()
        while True:
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    raise TimeoutError(
                        "timed out acquiring MME lock at {} after {}s".format(path, timeout)
                    )
                time.sleep(poll_interval)
        try:
            yield
        finally:
            try:
                fcntl.lockf(fd, fcntl.LOCK_UN)
            except OSError:
                pass


@contextmanager
def timeout(seconds=60):
    def _raise_timeout_error(signum, frame):
        raise Exception(408, "Timed out after {} seconds".format(seconds))

    try:
        signal.signal(signal.SIGALRM, _raise_timeout_error)
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)


class MultiModelException(Exception):
    def __init__(self, code, msg, pid):
        Exception.__init__(self, code, msg)
        self.pid = pid
        self.code = code
        self.msg = msg
