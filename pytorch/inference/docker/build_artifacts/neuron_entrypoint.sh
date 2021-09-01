#!/bin/bash

wait_for_nrtd() {
  nrtd_sock="/run/neuron.sock"
  SOCKET_TIMEOUT=300
  is_wait=true
  wait_time=0
  i=1
  sp="/-\|"
  echo -n "Waiting for neuron-rtd  "
  pid=$1
  while $is_wait; do
    if [ -S "$nrtd_sock" ]; then
      echo "$nrtd_sock Exist..."
      is_wait=false
    else
      sleep 1
      wait_time=$((wait_time + 1))
      if [ "$wait_time" -gt "$SOCKET_TIMEOUT" ]; then
        echo "neuron-rtd failed to start, exiting"
	      cat /tmp/nrtd.log
        exit 1
      fi
      printf "\b${sp:i++%${#sp}:1}"
    fi
  done
  cat /tmp/nrtd.log
}

python /usr/local/bin/deep_learning_container.py >> /dev/null &

nrtd_present=0
if [[ -z "${NEURON_RTD_ADDRESS}" ]]; then
  # Start neuron-rtd
  /opt/aws/neuron/bin/neuron-rtd -g unix:/run/neuron.sock --log-console  >>  /tmp/nrtd.log 2>&1 &
  nrtd_pid=$!
  echo "NRTD PID: "$nrtd_pid""
  #wait for nrtd to be up (5 minutes timeout)
  wait_for_nrtd $nrtd_pid
  export NEURON_RTD_ADDRESS=unix:/run/neuron.sock
  nrtd_present=1
else
  echo "Neuron RTD is running as a side car container...."
fi

neuron_monitor_running=0
if [[ ! -z "${NEURON_MONITOR_CW_REGION}" ]]; then
  # Start neuron monitor. If namespace/region variable is set then use it.
  if [[ ! -z "${NEURON_MONITOR_CONFIG_FILE}" ]]; then
    config="--config-file ${NEURON_MONITOR_CONFIG_FILE:+ $NEURON_MONITOR_CONFIG_FILE}"
  fi
  if [[ ! -z "${NEURON_MONITOR_CW_NAMESPACE}" ]]; then
    mnnamespace="--namespace ${NEURON_MONITOR_CW_NAMESPACE:+ $NEURON_MONITOR_CW_NAMESPACE}"
  fi
  region="--region ${NEURON_MONITOR_CW_REGION:+ $NEURON_MONITOR_CW_REGION}"
  /opt/aws/neuron/bin/neuron-monitor ${config:+ $config} | /opt/aws/neuron/bin/neuron-monitor-cloudwatch.py ${mnnamespace:+ $mnnamespace} ${region:+$region} >> /tmp/nm.log 2>&1 &
  nm_pid=$!
  echo "Neuron Monitor Started"
  neuron_monitor_running=1
fi

MODEL_STORE=/opt/ml/model
TS_CONFIG=/home/model-server/config.properties
MODEL_PATH=""

while getopts ":m:t:" opt; do
  case $opt in
    m) MODEL_PATH="$OPTARG"
    ;;
    t) TS_CONFIG="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

printf "Model path: %s\n" "$MODEL_PATH"
printf "TS_CONFIG: %s\n" "$TS_CONFIG"
# Start the Model Server
if [[ -z "$MODEL_PATH" ]]; then
  torchserve --start --ts-config /home/model-server/config.properties --model-store /opt/ml/model
else
  torchserve --start --ts-config $TS_CONFIG --models $MODEL_PATH
fi
status=$?
ts_pid=$(<"/home/model-server/tmp/.model_server.pid")
if [ $status -ne 0 ]; then
  echo "Failed to start TF Model Server: $status"
  exit $status
fi

# more than one service in a container. The container exits with an error
# if it detects that either of the processes has exited.
# Otherwise it loops forever, waking up every 60 seconds

while sleep 60; do
  if [ $nrtd_present -ne 0 ]; then
    ps aux |grep neuron-rtd |grep -q -v grep
    NRTD_STATUS=$?
    if [ $NRTD_STATUS -ne 0 ]; then
      echo "neuron-rtd service exited."
      cat /tmp/nrtd.log
      exit 1
    fi
  fi
  ps -p $ts_pid >/dev/null 2>&1
  MODEL_SERVER_STATUS=$?
  # If the torchserve pid exists then status would be 0
  # If not 0, then something is wrong
  if [ $MODEL_SERVER_STATUS -ne 0 ]; then
    echo "torchserve  has already exited."
    exit 1
  fi
  if [ $neuron_monitor_running -ne 0 ]; then
    if [ ! -d "/proc/${nm_pid}" ]; then
      echo "neuron-monitor is not running."
      cat /tmp/nm.log
    fi
  fi
done
