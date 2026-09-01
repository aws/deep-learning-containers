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
import errno
import fcntl
import logging
import threading
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)

MODEL_CONFIG_FILE = "/sagemaker/model-config.cfg"
DEFAULT_LOCK_FILE = "/sagemaker/lock-file.lock"

# In-process lock for greenlet serialization. After gevent.monkey.patch_all(),
# threading.Lock is greenlet-aware (gevent patches it). The file lock alone
# provides no mutual exclusion between greenlets in the same process because
# POSIX record locks are process-owned, not fd-owned.
_LOCAL = threading.Lock()

# Long-lived file handle — closing ANY fd on the lock file releases ALL of the
# process's locks on that file (POSIX semantics). Keep one fd open for the
# process lifetime.
_LOCK_FH = None


@contextmanager
def lock(path=DEFAULT_LOCK_FILE, poll_interval=0.05, timeout=60.0):
    """Greenlet-safe lock combining an in-process lock with a file lock.

    The in-process lock (threading.Lock, gevent-patched) serializes greenlets
    within the same worker process. The file lock serializes across workers
    when SAGEMAKER_GUNICORN_WORKERS > 1.

    fcntl.lockf is NOT gevent-patched — the blocking form parks the entire
    hub, stalling every greenlet in the worker including /ping. Use the
    non-blocking form in a retry loop with time.sleep (which IS patched)
    so /ping keeps serving during model-load contention.
    """
    global _LOCK_FH

    deadline = time.time() + timeout
    remaining = timeout

    if not _LOCAL.acquire(timeout=remaining):
        raise TimeoutError("timed out acquiring in-process MME lock after {}s".format(timeout))
    try:
        if _LOCK_FH is None:
            _LOCK_FH = open(path, "a+", encoding="utf8")

        fd = _LOCK_FH.fileno()
        while True:
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.time() >= deadline:
                    raise TimeoutError(
                        "timed out acquiring MME file lock at {} after {}s".format(path, timeout)
                    )
                time.sleep(poll_interval)
        try:
            yield
        finally:
            try:
                fcntl.lockf(fd, fcntl.LOCK_UN)
            except OSError as e:
                log.error("failed to release MME file lock: %s", e)
    finally:
        _LOCAL.release()


class MultiModelException(Exception):
    def __init__(self, code, msg, pid):
        Exception.__init__(self, code, msg)
        self.pid = pid
        self.code = code
        self.msg = msg
