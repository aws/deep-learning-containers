# TensorFlow benchmarking scripts

This folder contains the TF training scripts https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks.

## Basic usage
**execute_tensorflow_training.py train** uses SageMaker python sdk to start a training job. 

```bash
./execute_tensorflow_training.py train --help
Usage: execute_tensorflow_training.py train [OPTIONS] [SCRIPT_ARGS]...

Options:
  --framework-version [1.11.0|1.12.0]
                                  [required]
  --device [cpu|gpu]              [required]
  --py-versions TEXT
  --training-input-mode [File|Pipe]
  --networking-isolation / --no-networking-isolation
  --wait / --no-wait
  --security-groups TEXT
  --subnets TEXT
  --role TEXT
  --instance-counts INTEGER
  --batch-sizes INTEGER
  --instance-types TEXT
  --help                          Show this message and exit.

```
**execute_tensorflow_training.py generate_reports** generate benchmark reports. 

## Examples:

```bash
#!/usr/bin/env bash

./execute_tensorflow_training.py train \
--framework-version  1.11.0 \
--device gpu \
\
--instance-types ml.p3.2xlarge \
--instance-types ml.p3.8xlarge \
--instance-types ml.p3.16xlarge \
--instance-types ml.p2.xlarge \
--instance-types ml.p2.8xlarge \
--instance-types ml.p2.16xlarge \
\
--instance-counts 1 \
\
--py-versions py3 \
--py-versions py2 \
\
--subnets subnet-125fb674  \
\
--security-groups sg-ce5dd1b4  \
\
--batch-sizes 32 \
--batch-sizes 64 \
--batch-sizes 128 \
--batch-sizes 256 \
--batch-sizes 512 \
\
-- --model resnet32 --num_epochs 10 --data_format NHWC --summary_verbosity 1 --save_summaries_steps 10 --data_name cifar10
```

## Using other models, datasets and benchmarks configurations
```python tf_cnn_benchmarks/tf_cnn_benchmarks.py --help``` shows all the options that the script has.
