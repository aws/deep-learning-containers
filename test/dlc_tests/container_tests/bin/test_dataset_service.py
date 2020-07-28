import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

tfds.disable_progress_bar()

flags = tf.compat.v1.app.flags

flags.DEFINE_bool("local", False, "Run data service in process")
FLAGS = flags.FLAGS

def local_service():
    print("Starting Local Service")
    dispatcher = tf.data.experimental.service.DispatchServer(port=50050)
    dispatcher_address = dispatcher.target.split("://")[1]
    worker = tf.data.experimental.service.WorkerServer(
    port=0, dispatcher_address=dispatcher_address)
    print("Dispatcher target is ",dispatcher.target)
    return dispatcher, worker, dispatcher.target

def apply_transformations(ds_train):
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train

(ds_train, _), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    data_dir="/tmp/tfds",
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = apply_transformations(ds_train)
# Create dataset however you were before using the tf.data service.
dataset = ds_train
if(FLAGS.local):
    dispatcher, worker, service = local_service()
else:
    dispatcher_address = "localhost"
    dispatcher_port = "50050"
    service = "grpc://{}:{}".format(dispatcher_address, dispatcher_port)
# This will register the dataset with the tf.data service cluster so that
# tf.data workers can run the dataset to produce elements. The dataset returned
# from applying `distribute` will fetch elements produced by tf.data workers.
dataset = dataset.apply(tf.data.experimental.service.distribute(
    processing_mode="parallel_epochs", service=service))

for (x1,y1),(x2,y2) in zip(dataset, ds_train):
    np.allclose(x1,x2)
    np.allclose(y1,y2)

print("verified mnist dataset locally vs over service")
