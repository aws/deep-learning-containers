import os
import numpy as np
import time

def opt_in_opt_out_test():
  os.environ['TEST_MODE']='1'
  if os.path.exists("/tmp/test_request.txt"):
    os.system("rm /tmp/test_request.txt")
  os.environ["OPT_OUT_TRACKING"]="True"
  cmd = "python -c 'import torch'"
  os.system(cmd)
  time.sleep(1)
  assert(os.path.exists("/tmp/test_request.txt")==False)

  if os.path.exists("/tmp/test_request.txt"):
    os.system("rm /tmp/test_request.txt")
  os.environ["OPT_OUT_TRACKING"]="False"
  cmd = "python -c 'import torch'"
  os.system(cmd)
  time.sleep(1)
  assert(os.path.exists("/tmp/test_request.txt")==True)


  if os.path.exists("/tmp/test_request.txt"):
    os.system("rm /tmp/test_request.txt")
  os.environ["OPT_OUT_TRACKING"]="TRUE"
  cmd = "python -c 'import torch'"
  os.system(cmd)
  time.sleep(1)
  assert(os.path.exists("/tmp/test_request.txt")==False)

  if os.path.exists("/tmp/test_request.txt"):
    os.system("rm /tmp/test_request.txt")
  os.environ["OPT_OUT_TRACKING"]="true"
  cmd = "python -c 'import torch'"
  os.system(cmd)
  time.sleep(1)
  assert(os.path.exists("/tmp/test_request.txt")==False)


  if os.path.exists("/tmp/test_request.txt"):
    os.system("rm /tmp/test_request.txt")
  os.environ["OPT_OUT_TRACKING"]="XYgg"
  cmd = "python -c 'import torch'"
  os.system(cmd)
  time.sleep(1)
  assert(os.path.exists("/tmp/test_request.txt")==True)


  print("Opt-In/Opt-Out Test passed")

def perf_test():
  os.environ['TEST_MODE']='0'
  os.environ["OPT_OUT_TRACKING"]="False"
  NUM_ITERATIONS = 5

  for itr in range(NUM_ITERATIONS):
    total_time_in = 0
    for x in range(NUM_ITERATIONS):
      cmd = "python -c 'import time; start= time.time();import torch; '"
      start = time.time()
      os.system(cmd)
      total_time_in += time.time()-start
    print("avg out  time: ", total_time_in/NUM_ITERATIONS)

    total_time_out = 0
    for x in range(NUM_ITERATIONS):
      cmd = "export OPT_OUT_TRACKING='true' && python -c 'import time;start= time.time();import torch; '"
      start = time.time()
      os.system(cmd)
      total_time_out += time.time()-start
    print("avg out  time: ", total_time_out/NUM_ITERATIONS)

    np.testing.assert_allclose(total_time_in/NUM_ITERATIONS, total_time_out/NUM_ITERATIONS, rtol=0.2, atol=0.5)

    print("DLC Telemetry performance test Passed")

perf_test()
opt_in_opt_out_test()

print("All DLC telemetry test passed")
