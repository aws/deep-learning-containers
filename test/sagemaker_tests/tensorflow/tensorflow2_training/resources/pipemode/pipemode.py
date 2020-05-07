import json
import multiprocessing
import os
import tempfile

import tensorflow as tf
from sagemaker_tensorflow import PipeModeDataset

print("Starting estimator script")

ds = PipeModeDataset("elizabeth",benchmark=True)


class BenchmarkConfig(object):

    def __init__(self):
        self.hp = json.load(open('/opt/ml/input/config/hyperparameters.json'))

    @property
    def batch_size(self):
        return int(self.hp.get('batch_size', 5))

    @property
    def prefetch_size(self):
        return int(self.hp.get('prefetch_size', 1000))

    @property
    def channel(self):
        return self.hp.get('channel', 'elizabeth')

    @property
    def dimension(self):
        return int(self.hp['dimension'])

    @property
    def epochs(self):
        return int(self.hp.get('epochs', 3))

    @property
    def parallel_transform_calls(self):
        return int(self.hp.get('parallel_transform_calls', max(1, multiprocessing.cpu_count() - 2)))

    def __repr__(self):
        """Return all properties"""
        return str(vars(self))


config = BenchmarkConfig()


def input_fn():
    features = {
        'data': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.FixedLenFeature([], tf.int64),
    }

    def parse(record):
        parsed = tf.io.parse_single_example(serialized=record, features=features)
        return ({
            'data': tf.io.decode_raw(parsed['data'], tf.float64)
        }, parsed['labels'])

    ds = PipeModeDataset(config.channel)

    if config.epochs > 1:
        ds = ds.repeat(config.epochs)
    if config.prefetch_size > 0:
        ds = ds.prefetch(config.prefetch_size)
    ds = ds.map(parse, num_parallel_calls=config.parallel_transform_calls)
    ds = ds.batch(config.batch_size)
    return ds


# Perform Estimator training
column = tf.feature_column.numeric_column('data', shape=(config.dimension, ))
model_dir = tempfile.mkdtemp()
estimator = tf.estimator.LinearClassifier(feature_columns=[column])

print("About to call train")
estimator.train(input_fn=input_fn)

# Confirm that we have read the correct number of pipes
assert os.path.exists('/opt/ml/input/data/{}_{}'.format(config.channel, config.epochs + 1))

print("About to call evaluate")
result = estimator.evaluate(input_fn=input_fn)
for key, value in sorted(result.items()):
    print('%s: %s' % (key, value))


# Test that we can create a new PipeModeDataset after training has run
print("Validate that new PipeModeDataset on existing channel can be created")
tf.compat.v1.disable_eager_execution()

ds = PipeModeDataset(config.channel,benchmark=True)
with tf.compat.v1.Session() as sess:
    it = tf.compat.v1.data.make_one_shot_iterator(ds)
    next = it.get_next()
    sess.run(next)

print("Validate create, read, discard, recreate")

# Test that we can create a PipeModeDataset, discard it, and read from a new one
ds = PipeModeDataset(config.channel,benchmark=True)
with tf.compat.v1.Session() as sess:
    it = tf.compat.v1.data.make_one_shot_iterator(ds)
    next = it.get_next()


with tf.compat.v1.Session() as sess:
    it = tf.compat.v1.data.make_one_shot_iterator(ds)
    next = it.get_next()
    sess.run(next)

print("Validate recreate")
ds = PipeModeDataset(config.channel,benchmark=True)
