from __future__ import absolute_import
import subprocess


# run compat mounting by default
try:
    subprocess.run(["bash", "-m", "/usr/local/bin/start_cuda_compat.sh"])
except Exception as e:
    print(f"Error running script: {e}")
