#!/bin/bash

for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 4 --model sm-c5xl >> sm-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 8 --model sm-c5xl >> sm-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 16 --model sm-c5xl >> sm-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 32 --model sm-c5xl >> sm-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 4 --model sm-c518xl >> sm-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 8 --model sm-c518xl >> sm-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 16 --model sm-c518xl >> sm-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 32 --model sm-c518xl >> sm-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 64 --model sm-c518xl >> sm-perftest.log; done

for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 4 --model tfs-c5xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 8 --model tfs-c5xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 16 --model tfs-c5xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 32 --model tfs-c5xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 4 --model tfs-c518xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 8 --model tfs-c518xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 16 --model tfs-c518xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 32 --model tfs-c518xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 64 --model tfs-c518xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python perftest_endpoint.py --count 5000 --warmup 100 --workers 128 --model tfs-c518xl >> tfs-perftest.log; done


for i in $(seq 1 5); do python test/perf/perftest_endpoint.py --count 5000 --warmup 100 --workers 4 --model tfs-p2xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python test/perf/perftest_endpoint.py --count 5000 --warmup 100 --workers 8 --model tfs-p2xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python test/perf/perftest_endpoint.py --count 5000 --warmup 100 --workers 16 --model tfs-p2xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python test/perf/perftest_endpoint.py --count 5000 --warmup 100 --workers 32 --model tfs-p2xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python test/perf/perftest_endpoint.py --count 5000 --warmup 100 --workers 64 --model tfs-p2xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python test/perf/perftest_endpoint.py --count 5000 --warmup 100 --workers 4 --model tfs-p316xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python test/perf/perftest_endpoint.py --count 5000 --warmup 100 --workers 8 --model tfs-p316xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python test/perf/perftest_endpoint.py --count 5000 --warmup 100 --workers 16 --model tfs-p316xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python test/perf/perftest_endpoint.py --count 5000 --warmup 100 --workers 32 --model tfs-p316xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python test/perf/perftest_endpoint.py --count 5000 --warmup 100 --workers 64 --model tfs-p316xl >> tfs-perftest.log; done
for i in $(seq 1 5); do python test/perf/perftest_endpoint.py --count 5000 --warmup 100 --workers 128 --model tfs-p316xl >> tfs-perftest.log; done
