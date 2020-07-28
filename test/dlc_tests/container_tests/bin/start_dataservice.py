import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train_remote, _), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    data_dir="/tmp/tfds",
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

dispatcher = tf.data.experimental.service.DispatchServer(port=50050)
dispatcher_address = dispatcher.target.split("://")[1]
worker = tf.data.experimental.service.WorkerServer(
    port=0, dispatcher_address=dispatcher_address)
print("Starting Worker")
worker.join()
