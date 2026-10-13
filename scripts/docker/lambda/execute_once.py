"""Copied from the Lambda GPU workloads examples. Delete once the RIC ships pre-fork hooks."""

import os
import pickle
import time
from functools import wraps
from pathlib import Path


class ExecuteOnce:
    """
    Ensures a function is executed exactly once across all forked processes
    during Lambda's INIT phase. Uses atomic locking and result caching in /dev/shm.

    Args:
        func: Function to execute once.
        lock_path: Path to lock file in /dev/shm (e.g., "/dev/shm/my_func.lock").
        result_path: Path to result file in /dev/shm (e.g., "/dev/shm/my_func.result").
        timeout: Max wait time for result (seconds). Default: 30.
        serializer: Function to serialize result (default: pickle.dump).
        deserializer: Function to deserialize result (default: pickle.load).
    """

    def __init__(
        self,
        func,
        lock_path: str,
        result_path: str,
        timeout: float = 30.0,
        serializer=pickle.dump,
        deserializer=pickle.load,
    ):
        self.func = func
        self.lock_path = lock_path
        self.result_path = result_path
        self.timeout = timeout
        self.serializer = serializer
        self.deserializer = deserializer

    def __call__(self, *args, **kwargs):
        """Execute the function once or wait for cached result."""
        if not Path(self.result_path).exists():
            try:
                # Atomic lock acquisition
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL)
                try:
                    # Execute function and cache result
                    result = self.func(*args, **kwargs)
                    with open(self.result_path, "wb") as f:
                        self.serializer(result, f)
                    return result
                finally:
                    os.close(fd)
                    os.unlink(self.lock_path)  # Cleanup lock
            except FileExistsError:
                # Another process has the lock; wait for result
                start_time = time.time()
                while not Path(self.result_path).exists():
                    if time.time() - start_time > self.timeout:
                        raise TimeoutError(f"Result not available within {self.timeout}s")
                    time.sleep(0.05)  # Non-blocking wait

        # Read cached result
        with open(self.result_path, "rb") as f:
            return self.deserializer(f.read())


def execute_once(
    lock_path: str,
    result_path: str,
    timeout: float = 30.0,
    serializer=pickle.dump,
    deserializer=pickle.load,
):
    """Decorator version of ExecuteOnce."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            executor = ExecuteOnce(
                func=func,
                lock_path=lock_path,
                result_path=result_path,
                timeout=timeout,
                serializer=serializer,
                deserializer=deserializer,
            )
            return executor(*args, **kwargs)

        return wrapper

    return decorator
