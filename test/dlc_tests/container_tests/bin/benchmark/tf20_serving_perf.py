# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for tensorflow_model_server."""

import atexit
import os
import shlex
import socket
import subprocess
import sys
import time
import pickle
import shutil
import boto3
import botocore
import marshal
import argparse
import logging
import pprint
from multiprocessing.dummy import Pool

# This is a placeholder for a Google-internal import.

import grpc
from grpc.beta import implementations
from grpc.beta import interfaces as beta_interfaces
from grpc.framework.interfaces.face import face
import tensorflow as tf
import numpy as np

from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import flags
from tensorflow_serving.apis import classification_pb2
# from tensorflow_serving.apis import get_model_status_pb2
# from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.estimator import model_fn as model_fn_lib

FLAGS = flags.FLAGS

RPC_TIMEOUT = 600.0
CHANNEL_WAIT_TIMEOUT = 5.0
WAIT_FOR_SERVER_READY_INT_SECS = 600
NUM_PREDICTIONS = 5

SERVING_OUPUT_LOG = "serving_output.log"

DOCKER_NAME = "tensorflow_inference_container"


def print_helper(cmd):
    print("****************************************")
    print(cmd)
    print("****************************************")


def download_test_image():
    if not os.path.isfile("/tmp/data/angeldog.jpg"):
        os.system(
            "wget http://arumi.blog.kataweb.it/files/photos/uncategorized/2007/05/22/angeldog.jpg")
        os.system("mkdir -p /tmp/data")
        os.system("mv angeldog.jpg /tmp/data")
        os.system("rm angeldog.jpg")


def PickUnusedPort():
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def WaitForServerReady(port):
    """Waits for a server on the localhost to become ready."""
    for _ in range(0, WAIT_FOR_SERVER_READY_INT_SECS):
        time.sleep(1)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'intentionally_missing_model'

        try:
            # Send empty request to missing model
            channel = grpc.insecure_channel('localhost:{}'.format(port))
            stub = prediction_service_pb2_grpc.PredictionServiceStub(
                channel)
            stub.Predict(request, RPC_TIMEOUT)
        except grpc.RpcError as error:
            # Missing model error will have details containing 'Servable'
            if 'Servable' in error.details():
                print('Server is ready')
                break


class TensorflowModelServerTester(object):
    """This class defines integration test cases for tensorflow_model_server."""

    def __init__(self, ServingInput, model_server=None, concurrent=1):
        """ServingInput is a 4 item tuple.
        0: Model Name
        1: Model Path
        2: Signature Name
        3: Predict_Input_fn
        """
        self.Predict_Input_fn = ServingInput[3]
        self.model_name = ServingInput[0]
        self.model_path = ServingInput[1]
        self.sig_name = ServingInput[2]
        self.server_proc = None
        self.concurrent = concurrent
        if (model_server != None):
            self.binary = model_server
            if (not os.path.isfile(self.binary)):
                print("Can't find Tensorflow Serving Binary at %s please point to TFS binary" % (self.binary))
                exit(1)
        else:
            self.binary = None
        self.open_procs = []

    def TerminateProcs(self):
        """Terminate all processes."""
        print('Terminating all processes...')
        if self.server_proc is not None:
            print(self.server_proc)
            self.output_file.close()
            self.server_proc.terminate()

    def RunServer(self,
                  port,
                  model_name,
                  model_path,
                  batching_parameters_file='',
                  grpc_channel_arguments='',
                  wait_for_server_ready=True):
        """Run tensorflow_model_server using test config."""
        print('Starting test server...')

        command = 'docker' if processor == 'cpu' else 'nvidia-docker'

        env_command = ' -e TENSORFLOW_INTER_OP_PARALLELISM=2 -e TENSORFLOW_INTRA_OP_PARALLELISM=72 -e KMP_AFFINITY=\'granularity=fine,verbose,compact,1,0\' -e OMP_NUM_THREADS=36 -e TENSORFLOW_SESSION_PARALLELISM=9 -e KMP_BLOCKTIME=1 -e KMP_SETTINGS=0 ' if processor == 'cpu' else ''

        command += ' run --rm --name ' + DOCKER_NAME + env_command + ' -p 8500:8500 -v /home/ubuntu/src/container_tests:/test --mount type=bind,source=' + model_path + ',target=/models/' + model_name + ' -e MODEL_NAME=' + model_name + ' -itd ' + docker_image_name
        port = 8500

        print_helper(command)
        my_env = os.environ.copy()
        self.output_file = open(SERVING_OUPUT_LOG, 'w')
        self.server_proc = subprocess.Popen(shlex.split(command), env=my_env, stdout=self.output_file)
        self.open_procs.append(self.server_proc)
        print('Server started')
        if wait_for_server_ready:
            WaitForServerReady(port)
        return 'localhost:' + str(port)

    def _Predict(self, model_server_address, request_timeout=30, iterations=NUM_PREDICTIONS):
        """Helper method to call predict on models we want to test
        input_fn: This will return 4 lists
        [input_names], [input_data], [input_shapes], [input_types]
        model_name = name of model testing
        signature_name = default to SERVING
        """
        print("Sending Predict request...")
        host, port = model_server_address.split(':')
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = self.sig_name
        input_names, input_data, input_shapes, input_types = self.Predict_Input_fn()

        # input_names, input_data, input_shapes, input_types = self.convert_to_proto_artifacts(self.Predict_Input_fn())

        for ii in range(len(input_names)):  # Goverened by the input_names'
            print(input_shapes)
            if (input_types[ii] != None):
                request.inputs[input_names[ii]].CopyFrom(
                    tf.compat.v1.make_tensor_proto(input_data[ii], shape=input_shapes[ii], dtype=input_types[ii]))
            else:
                request.inputs[input_names[ii]].CopyFrom(
                    tf.compat.v1.make_tensor_proto(input_data[ii], shape=input_shapes[ii]))

        # Create the stub and channel
        channel = grpc.insecure_channel(model_server_address)
        timing = []
        stub = prediction_service_pb2_grpc.PredictionServiceStub(
            channel)
        p = Pool(self.concurrent)

        def do_pred(x):
            start = time.time()
            result = stub.Predict(request, request_timeout)
            end = time.time()
            return (result, end - start)

        for each in range(iterations):
            res = p.map(do_pred, range(self.concurrent))
            result = res[-1][0]
            times = [x[1] for x in res]
            timing.append((np.min(times), np.mean(times), np.max(times)))
        results = {}

        for output in result.outputs:
            results[output] = (tf.compat.v1.make_ndarray(
                result.outputs[output]))

        return results, timing

    def testServing(self, iterations=2):
        """ServingInput is a 4 item tuple.
        0: Model Name
        1: Model Path
        2: Signature Name
        3: Predict_Input_fn
        """
        # atexit.register(self.TerminateProcs)

        model_server_address = self.RunServer(
            PickUnusedPort(),
            self.model_name,
            self.model_path,
        )
        result = self._Predict(model_server_address, RPC_TIMEOUT, iterations)
        print("Terminating Proc")
        self.server_proc.terminate()
        self.server_proc.wait()
        os.system("docker rm -f {}".format(DOCKER_NAME))
        return result


def test_serving(ServingTests, binary=None, concurrent=1, exclude_list=[], iterations=2):
    results = {}
    for each in ServingTests:
        if (type(each) == list):
            if (each[0] not in exclude_list):
                results[each[0]] = {}
                tester = TensorflowModelServerTester(each, binary, concurrent)
                res = tester.testServing(iterations=iterations)
                results[each[0]]["values"] = res[0]
                results[each[0]]["timing"] = res[1]
        else:
            if (ServingTests[0] not in exclude_list):
                results[ServingTests[0]] = {}
                tester = TensorflowModelServerTester(ServingTests, binary, concurrent)
                res = tester.testServing(iterations=iterations)
                results[ServingTests[0]]["values"] = res[0]
                results[ServingTests[0]]["timing"] = res[1]
            break

    return results


def download_file(bucket_name, s3, key):
    working_dir = "/tmp/test_eia_serving_"
    if (not os.path.isdir(working_dir)):
        os.mkdir(working_dir)

    fname = key + ".zip"

    working_dir = working_dir + "/" + key  # WORKING PLUS KEY
    print(key)

    if (os.path.isdir(working_dir)):
        shutil.rmtree(working_dir)
    os.mkdir(working_dir)

    destfile = working_dir + "/" + fname
    print("destfile", destfile)
    try:
        s3.Bucket(bucket_name).download_file(fname, destfile)

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("Object does not exist")
        else:
            raise

    if destfile.endswith(".zip"):
        os.system("unzip " + destfile + " -d /tmp/test_eia_serving_/")

    with open("%s/metadata.pkl" % (working_dir), "rb") as f2:
        metadata_read = f2.read()

        # First check python version
    if sys.version_info >= (3, 0):
        metadata = pickle.loads(metadata_read, encoding='latin1')
    else:
        metadata = pickle.loads(metadata_read)

    # Process input fn
    from types import FunctionType
    input_fn = metadata["input_fn"]

    exec(input_fn[0])
    id2 = eval(input_fn[1])

    metadata["input_fn"] = id2

    model_path = ''

    if os.path.isdir(working_dir + "/" + key):
        model_path = working_dir + "/" + key
    else:
        model_path = working_dir + "/" + key + ".pb"

    if not model_path:
        sys.exit("No model found in directory")
    output = [metadata['Test name'], model_path,
              metadata['sig_name'], metadata['input_fn']]
    return output


def upload_files(ServingTests, bucket_name):
    for each in ServingTests:
        print(each)
        upload_file(each, bucket_name)


def upload_file(ServingTest, bucket_name):
    # holds metadata i.e. classes, input names, output names
    working_dir = "/tmp/test_eia_serving_"
    # If clean up hasn't happened
    if (os.path.isdir(working_dir)):
        shutil.rmtree(working_dir)
    os.mkdir(working_dir)
    pickle_dict = {}
    zip_filename = ServingTest[0]
    model_file = ServingTest[1]
    working_dir = working_dir + "/" + zip_filename
    os.mkdir(working_dir)

    def is_pb_in_dir(dir_name):

        for versions in os.listdir(dir_name):
            for fname in os.listdir(dir_name + "/" + versions):
                if fname.endswith(".pb"):
                    print("Found pb")

                    return (True and "variables" in os.listdir(dir_name + "/" + versions))
        return False

    if not os.path.isfile(model_file):
        if (not (os.path.isdir(model_file) or is_pb_in_dir(model_file))):
            sys.exit("Invalid model file name")

    input_fn = ServingTest[3]

    pickle_dict["Test name"] = ServingTest[0]
    pickle_dict["sig_name"] = ServingTest[2]
    # Need to modify the pickling of a function.
    # Process input_fn
    import inspect
    funcdetails = [

        inspect.getsource(input_fn),
        input_fn.__name__,
    ]
    input_fn = funcdetails
    pickle_dict['input_fn'] = input_fn

    def copyfile(file_or_dir):
        if (os.path.isdir(working_dir)):
            shutil.copytree(file_or_dir, working_dir + "/" + zip_filename)
        else:
            shutil.copyfile(file_or_dir, working_dir + "/" + zip_filename + ".pb")

    # Copy in the model file or directory
    if (model_file.endswith("/")):
        model_file = model_file[:-1]
    copyfile(model_file)

    pickle.dump(
        pickle_dict, open(working_dir + "/metadata.pkl", "wb"), 2)

    # zips file together
    os.chdir("/tmp/test_eia_serving_")
    os.system("zip -r " + zip_filename + " " + zip_filename)
    # uploads zip file to s3
    os.system("aws s3 cp " + zip_filename + ".zip" + " s3://" + bucket_name)


def mnist_input_fn():
    input_names = ["images"]
    np.random.seed(0)
    input_data = np.random.rand(1, 784)
    input_shapes = [[1, 784]]
    input_types = [(tf.float32)]
    return input_names, input_data, input_shapes, input_types


def saved_model_half_plus_two_input_fn():
    input_names = ["x"]
    np.random.seed(0)
    input_data = np.random.rand(1, 1)
    input_shapes = [[1, 1]]
    input_types = [(tf.float32)]
    return input_names, input_data, input_shapes, input_types


def Inception_input_fn():
    f = open("/tmp/data/angeldog.jpg", 'rb')
    input_names = ['images']
    input_shapes = [[1]]
    input_data = [f.read()]
    input_types = [None]
    f.close()
    return input_names, input_data, input_shapes, input_types


def Resnet_50_v1_input_fn():
    input_names = ['input']
    np.random.seed(0)
    input_data = np.random.rand(1, 224, 224, 3)
    input_shapes = [(1, 224, 224, 3)]
    input_types = [tf.float32]
    return input_names, input_data, input_shapes, input_types


def Resnet_input_fn():
    input_names = ['input']
    np.random.seed(0)
    input_data = np.random.rand(128, 224, 224, 3)
    input_shapes = [(128, 224, 224, 3)]
    input_types = [tf.float32]
    return input_names, input_data, input_shapes, input_types


def Resnet101_input_fn():
    input_names = ['inputs']
    np.random.seed(0)
    input_data = np.random.rand(1, 224, 224, 3)
    input_shapes = [(1, 224, 224, 3)]
    input_types = [tf.uint8]
    return input_names, input_data, input_shapes, input_types


def resnet50v2_taehoon_input_fn():
    input_names = ['Placeholder:0']
    np.random.seed(0)
    input_data = np.random.rand(1, 224, 224, 3)
    input_shapes = [(1, 224, 224, 3)]
    input_types = [tf.float32]
    return input_names, input_data, input_shapes, input_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="Run a particular model locally first then s3")
    parser.add_argument("--run_s3", help="Run a particular model from s3")
    parser.add_argument(
        "--upload", help="Upload to S3, will invoke interactive mode")
    parser.add_argument(
        "--run_all_s3", help="Run all models defined by Keys", action="store_true")
    parser.add_argument(
        "--upload_all", help="Upload all models in model_info definition", action="store_true")
    parser.add_argument(
        "--binary", default="/usr/bin/tensorflow_model_server", help="Where tensorflow_model_server")
    parser.add_argument(
        "--concurrent", help="Number of concurrent threads (GIL limited)", default=1, type=int)
    parser.add_argument(
        "--get_perf", help="Get Performance statistics", action="store_true")
    parser.add_argument(
        "--iterations", help="Run iterations", type=int, default=2)
    parser.add_argument(
        "--validate_gpu_binary", help="Run GPU Test to validate binary is a GPU binary", action="store_true")
    parser.add_argument(
        "--validate_mkl_binary", help="Run MKL Test to validate binary is a MKL binary", action="store_true")
    parser.add_argument(
        "--docker_image_name", help="The docker image name used to create a container")
    parser.add_argument(
        "--processor", help="Run on gpu or cpu", default='gpu')
    parser.add_argument(
        "--exclude", help="Exclude certain models when running all s3 models", nargs="+")

    # name of S3 bucket to be accessed
    bucket_name = "tf-test-models"
    s3 = boto3.resource('s3')
    args = parser.parse_args()
    download_test_image()

    ServingTests = []
    # ServingTests.append(["saved_model_half_plus_two_gpu", "/home/ubuntu/saved_model_half_plus_two_gpu/", 'serving_default', saved_model_half_plus_two_input_fn])
    ServingTests.append(
        ["saved_model_half_plus_two_cpu", "/home/ubuntu/saved_model_half_plus_two_cpu/", 'serving_default',
         saved_model_half_plus_two_input_fn])

    ServingTests.append(
        ["INCEPTION", "/home/ubuntu/s3model/serving/SERVING_INCEPTION/SERVING_INCEPTION", 'predict_images',
         Inception_input_fn])
    ServingTests.append(
        ["MNIST", "/home/ubuntu/s3model/serving/SERVING_MNIST/SERVING_MNIST", 'predict_images', mnist_input_fn])
    ServingTests.append(
        ["Resnet50v2", "/home/ubuntu/s3model/serving/SERVING_Resnet-50-v2/SERVING_Resnet-50-v2/", 'serving_default',
         Resnet_50_v1_input_fn])
    ServingTests.append(
        ["RCNN-Resnet101-kitti", "/home/ubuntu/s3model/serving/SERVING_RCNN-Resnet101/SERVING_RCNN-Resnet101",
         'serving_default', Resnet101_input_fn])
    ServingTests.append(
        ["SSDResnet50Coco", "/home/ubuntu/s3model/serving/SERVING_ssd_resnet50_v1_coco/SERVING_ssd_resnet50_v1_coco",
         'serving_default', Resnet101_input_fn])
    ServingTests.append(["resnet50v2-taehoon", "/home/ubuntu/s3model/serving/resnet50-taehoon/", 'serving_default',
                         resnet50v2_taehoon_input_fn])

    keys = [
        # "saved_model_half_plus_two_cpu",
        "INCEPTION",
        "MNIST",
        "Resnet50v2",
        #"RCNN-Resnet101-kitti",
        #"SSDResnet50Coco",
    ]
    if args.exclude:
        keys = [x for x in keys if x not in args.exclude]

    if (os.path.isdir("/tmp/downloads")):
        os.system("rm -rf /tmp/downloads/*")
    else:
        os.system("mkdir /tmp/downloads")

    if (args.docker_image_name):
        global docker_image_name
        docker_image_name = args.docker_image_name

    global processor
    processor = args.processor

    if (not args.binary):
        print("No binary specified, Using Default binary in PATH")
        args.binary = None

    keynames = [x[0] for x in ServingTests]
    if (args.upload):
        if (args.upload in keynames):
            print("Uploading %s" % (args.upload))
            upload_file(ServingTests[keynames.index(args.upload)], bucket_name)
            print("finished Uploading %s" % (args.upload))
        return
    elif (args.run):
        dir_path = None
        if (args.run in keynames):
            dir_path = ServingTests[keynames.index(args.run)][1]
        if (args.run in keynames and (os.path.isdir(dir_path))):
            ServingTests = [ServingTests[keynames.index(args.run)]]
            print("Found a local Model for %s" % (args.run))
        else:
            print("Downloading S3 Model [%s]" % (args.run))
            output = download_file(bucket_name, s3, args.run)
            ServingTests = []
            ServingTests.append(output)
    elif (args.run_s3):
        print("Downloading S3 Model [%s]" % (args.run_s3))
        output = download_file(bucket_name, s3, args.run_s3)
        ServingTests = []
        ServingTests.append(output)
    elif (args.upload_all):
        upload_files(ServingTests, bucket_name)
        print("Finished uploading models [%s]" % (", ".join(keynames)))
        return
    elif (args.validate_gpu_binary):
        # If there are more models that need to be added for DLAMI, it should be added here.
        keys = [
            "saved_model_half_plus_two_gpu"
        ]
        ServingTests = []
        for key in keys:
            print("Downloading %s" % (key))
            output = download_file(bucket_name, s3, key)
            ServingTests.append(output)
            print(output)
    elif (args.validate_mkl_binary):
        # If there are more models that need to be added for DLAMI, it should be added here.

        os.environ["MKL_VERBOSE"] = str(1)
        keys = [
            "INCEPTION"
        ]
        ServingTests = []
        for key in keys:
            print("Downloading %s" % (key))
            output = download_file(bucket_name, s3, key)
            ServingTests.append(output)
            print(output)

    elif (args.run_all_s3):
        if (len(keys) > 0):
            print("Running models [%s]" % (", ".join(keys)))
        else:
            print("No models to run, exiting")
            return
        ServingTests = []
        for key in keys:
            print("Downloading %s" % (key))
            output = download_file(bucket_name, s3, key)
            ServingTests.append(output)
            print(output)
    else:
        # elif(args.run_all): This is the DEFAULT
        # input names of files to download
        parser.print_help()
        exit()

    if (args.iterations):
        iterations = args.iterations
    else:
        iterations = 2
    server_results = {}
    server_results = test_serving(ServingTests, args.binary, args.concurrent, iterations=iterations)
    if (args.get_perf):
        server_results = get_perf(server_results)

    print("Printing server timing")
    printresults(server_results)

    if (args.validate_mkl_binary):
        fp = open(SERVING_OUPUT_LOG)
        for line in fp.readlines():
            if ('MKL_VERBOSE Intel' in line):
                print("This is an MKL Binary")
                print("Proof: ", line)
                break
        else:
            raise ModuleNotFoundError("This isn't an MKL binary")


def get_perf(results):
    for model in results:
        stats = {}
        min_results = []
        mean_results = []
        max_results = []
        for each in results[model]['timing']:
            min_results.append(each[0])
            mean_results.append(each[1])
            max_results.append(each[2])

        if (len(mean_results) > 1):
            mean_results = mean_results[1:]  # REmove the first inference

        stats['p50'] = np.percentile(mean_results, 50, interpolation='nearest')
        stats['p90'] = np.percentile(mean_results, 90, interpolation='nearest')
        stats['p99'] = np.percentile(mean_results, 99, interpolation='nearest')
        stats['mean'] = np.mean(mean_results)
        stats['min'] = np.min(mean_results)
        stats['max'] = np.max(mean_results)
        results[model]['timing'] = stats
    return results


def printresults(results):
    for model in results:
        line = "%20s : " % (model)
        if (type(results[model]['timing']) == dict):
            for stat in results[model]['timing']:
                line += "%s : %6.4f ," % (stat, results[model]['timing'][stat])
        else:
            for each in results[model]['timing']:
                line += "[%6.4f,%6.4f,%6.4f]," % (each)
        print(line)


if __name__ == '__main__':
    main()
