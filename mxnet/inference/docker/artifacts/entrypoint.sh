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

MODEL_STORE=/opt/ml/model
MMS_CONFIG=/home/model-server/config.properties
MODEL_PATH=""

while getopts ":m:t:" opt; do
  case $opt in
    m) MODEL_PATH="$OPTARG"
    ;;
    t) MMS_CONFIG="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

printf "Model path: %s\n" "$MODEL_PATH"
printf "MMS_CONFIG: %s\n" "$TS_CONFIG"
# Start the Model Server
if [[ -z "$MODEL_PATH" ]]; then
  multi-model-server --start --mms-config /home/model-server/config.properties --model-store /opt/ml/model &
else
  multi-model-server --start --mms-config $MMS_CONFIG --models $MODEL_PATH &
fi
status=$?
if [ $status -ne 0 ]; then
  echo "Failed to start MMS Model Server: $status"
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
  ps aux |grep mms |grep -q -v grep
  MODEL_SERVER_STATUS=$?
  # If the greps above find anything, they exit with 0 status
  # If they are not both 0, then something is wrong
  if [ $MODEL_SERVER_STATUS -ne 0 ]; then
    echo "multi-model-server  has already exited."
    exit 1
  fi
done
