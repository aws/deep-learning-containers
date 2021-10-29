#!/bin/bash

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