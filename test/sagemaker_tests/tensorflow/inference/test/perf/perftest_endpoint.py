# Copyright 2018-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import argparse
import multiprocessing
import sys
import time

import boto3


class PerfTester(object):
    def __init__(self):
        self.engine = None
        self.count = None
        self.payload_kb = None
        self.start_time = None
        self.end_time = None

    def test_worker(self, id, args, count, test_data, error_counts):
        client = boto3.client('sagemaker-runtime')

        endpoint_name = test_data[0]
        data = test_data[1]
        for i in range(count):
            try:
                response = client.invoke_endpoint(EndpointName=endpoint_name,
                                                  Body=data,
                                                  ContentType='application/json',
                                                  Accept='application/json',
                                                  CustomAttributes='tfs-model-name=cifar')
                _ = response['Body'].read()
            except:
                error_counts[id] += 1

    def test(self, args, count, test_data):
        self.count = args.count * args.workers
        self.payload_kb = len(test_data[1]) / 1024.0

        manager = multiprocessing.Manager()
        error_counts = manager.dict()
        workers = []
        for i in range(args.workers):
            error_counts[i] = 0
            w = multiprocessing.Process(target=self.test_worker,
                                        args=(i, args, count, test_data, error_counts))
            workers.append(w)

        self.start_time = time.time()
        for w in workers:
            w.start()

        for w in workers:
            w.join()

        self.errors = sum(error_counts.values())
        self.end_time = time.time()

    def report(self, args):
        elapsed = self.end_time - self.start_time
        report_format = '{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{:.3f}'
        report = report_format.format(args.model,
                                      args.workers,
                                      self.count / elapsed,
                                      ((elapsed / args.count) * 1000),
                                      (self.payload_kb * self.count) / elapsed / 1024,
                                      elapsed,
                                      self.count,
                                      self.errors,
                                      self.payload_kb)
        print('model,workers,r/s,ms/req,mb/s,total s,requests,errors,payload kb')
        print(report)

    def parse_args(self, args):
        parser = argparse.ArgumentParser('performance tester')
        parser.set_defaults(func=lambda x: parser.print_usage())
        parser.add_argument('--count', help='number of test iterations', default=1000, type=int)
        parser.add_argument('--warmup', help='number of warmup iterations', default=100, type=int)
        parser.add_argument('--workers', help='number of model workers (and clients)', default=1,
                            type=int)
        parser.add_argument('--model', help='model id', default='half_plus_three')
        return parser.parse_args(args)

    def run(self, args):
        args = self.parse_args(args)
        test_data = TEST_DATA[args.model]
        self.test(args, min(args.warmup, args.count), test_data)
        self.test(args, args.count, test_data)
        self.report(args)


def _read_file(path):
    with open(path, 'rb') as f:
        return f.read()


def _random_payload(size_in_kb):
    return bytes(bytearray(size_in_kb * 1024))


# add/change these to match your endpoints
TEST_DATA = {
    'sm-p2xl': ('sagemaker-tensorflow-2018-11-03-14-38-51-707', b'[' + _read_file('test/resources/inputs/test-cifar.json') + b']'),
    'sm-p316xl': ('sagemaker-tensorflow-2018-11-03-14-38-51-706', b'[' + _read_file('test/resources/inputs/test-cifar.json') + b']'),
    'tfs-p2xl': ('sagemaker-tfs-p2-xlarge', _read_file('test/resources/inputs/test-cifar.json')),
    'tfs-p316xl': ('sagemaker-tfs-p3-16xlarge', _read_file('test/resources/inputs/test-cifar.json')),
    'tfs-c5xl': ('sagemaker-tfs-c5-xlarge', _read_file('test/resources/inputs/test-cifar.json')),
    'tfs-c518xl': ('sagemaker-tfs-c5-18xlarge', _read_file('test/resources/inputs/test-cifar.json')),
    'sm-c5xl': ('sagemaker-tensorflow-cifar-c5.xlarge', b'[' + _read_file('test/resources/inputs/test-cifar.json') + b']'),
    'sm-c518xl': ('sagemaker-tensorflow-cifar-c5.18xlarge', b'[' + _read_file('test/resources/inputs/test-cifar.json') + b']')
}

if __name__ == '__main__':
    PerfTester().run(sys.argv[1:])
