import tensorflow as tf
import horovod.tensorflow.keras as hvd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_type")
args = parser.parse_args()
data_type = args.data_type

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(128)

if data_type == "AMP":
    # if this code block get excecuted, data type is mixed-precision AKA AMP. If not, it is FP32.
    # When ampere comes and TF32 is available, we need to extend the test to run TF32 as well
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', 128)
    tf.keras.mixed_precision.experimental.set_policy(policy)
elif data_type != "FP32":
    raise Exception("not supported data type.\n")

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Horovod: adjust learning rate based on number of GPUs.
opt = tf.optimizers.Adam(0.001)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                    optimizer=opt,
                    metrics=['accuracy'],
                    experimental_run_tf_function=False)

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Train the model.
# Horovod: adjust number of steps based on number of GPUs.
mnist_model.fit(
    dataset,
    steps_per_epoch=500 // hvd.size(),
    callbacks=callbacks,
    epochs=24,
    verbose=1 if hvd.rank() == 0 else 0
)
