#!/bin/bash

ab -k -n 10000 -c 16 -p test/resources/inputs/test.json -T 'application/json' http://localhost:8080/tfs/v1/models/half_plus_three:predict
ab -k -n 10000 -c 16 -p test/resources/inputs/test.json -T 'application/json' http://localhost:8080/invocations
ab -k -n 10000 -c 16 -p test/resources/inputs/test.jsons -T 'application/json' http://localhost:8080/invocations
ab -k -n 10000 -c 16 -p test/resources/inputs/test.csv -T 'text/csv' http://localhost:8080/invocations
ab -k -n 10000 -c 16 -p test/resources/inputs/test-cifar.json -T 'application/json'     -H 'X-Amzn-SageMaker-Custom-Attributes: tfs-model-name=cifar' http://localhost:8080/invocations

# Larger payloads are generated and removed when this script exits.
TEMPFILE='/tmp/perftest_data'
trap 'rm -f $TEMPFILE' EXIT

echo "Generating data"
# Creates a 10MB file with 10000 columns per line.
python test/perf/data_generator.py -c 'text/csv' -s 10000 -p 10 -u MB > $TEMPFILE || exit $?
ab -k -n 10 -c 1 -p "$TEMPFILE" -T 'text/csv' http://localhost:8080/invocations

python test/perf/data_generator.py -c 'application/json' -s 10000 -p 10 -u MB > $TEMPFILE || exit $?
ab -k -n 10 -c 1 -p "$TEMPFILE" -T 'application/json' http://localhost:8080/invocations

python test/perf/data_generator.py -c 'application/jsonlines' -s 10000 -p 10 -u MB > $TEMPFILE || exit $?
ab -k -n 10 -c 1 -p "$TEMPFILE" -T 'application/jsonlines' http://localhost:8080/invocations
