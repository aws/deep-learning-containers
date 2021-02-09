# Standard Library
import errno
import os
import shutil
import time

# Third Party
import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten
from utils import log_result

# First Party
import smdistributed.modelparallel.tensorflow as smp

tf.random.set_seed(1234)

cfg = {
    "microbatches": 2,
    "horovod": True,
    "placement_strategy": "spread",
    "partitions": 2,
    "xla": False,
    "pipeline": "interleaved",
    "optimize": "speed",
}
smp.init(cfg)

cache_dir = os.path.join(os.path.expanduser("~"), ".keras", "datasets")
if not os.path.exists(cache_dir):
    try:
        os.mkdir(cache_dir)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
            pass
        else:
            raise

# Download and load MNIST dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
    "MNIST-data-%d" % smp.rank()
)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000, seed=123 + smp.dp_rank())
    .batch(32)
)


class MyModel(smp.DistributedModel):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(
            32, 3, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=12)
        )
        self.conv0 = Conv2D(
            32, 3, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=12)
        )
        self.flatten = Flatten()
        self.d1 = Dense(
            128, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=192)
        )
        self.d2 = Dense(10, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=126))
        self.bn = BatchNormalization()

    def first(self, x):
        with tf.name_scope("first"):
            x = self.conv1(x)
            # x = self.bn(x)
            return self.flatten(x)

    def second(self, x):
        with tf.name_scope("second"):
            x = self.d1(x)
            return self.d2(x)

    def call(self, x):
        x = self.first(x)
        return self.second(x)


# Create an instance of the model
model = MyModel()

# Profile the model for auto-partition
step = 0
for images, labels in train_ds:
    model.profile(images)
    step += 1
    if step == 2:
        break

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name="train_loss")


train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")


@smp.step(
    input_signature=[
        tf.TensorSpec(shape=[32, 28, 28, 1], dtype=tf.float64),
        tf.TensorSpec(shape=[32], dtype=tf.uint8),
        tf.TensorSpec(shape=[], dtype=tf.bool),
    ],
    non_split_inputs=["first_batch"],
)
def get_grads(images, labels, first_batch):
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)

    gradients = optimizer.get_gradients(loss, model.trainable_variables)

    return gradients, predictions, loss


@tf.function
def train_step(images, labels, first_batch):
    gradients, predictions, loss = get_grads(images, labels, first_batch)
    gradients = [hvd.allreduce(g.reduce_mean()) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    train_loss(loss.reduce_mean())
    train_accuracy(labels, predictions.merge())
    return loss.reduce_mean()


t0 = None
step = 0
for epoch in range(5):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()

    for batch, (images, labels) in enumerate(train_ds):
        if step == 1:
            t0 = time.time()
        train_step(images, labels, tf.constant(batch == 0))
        step += 1

    print(
        "Epoch {}, Accuracy: {}, Loss: {}".format(
            epoch + 1, train_accuracy.result(), train_loss.result()
        )
    )

t1 = time.time()
log_result("Time-to-train", t1 - t0)
assert t1 - t0 < 220.0

save_path = "./hvd2_conv_saved_model_multinode"
model.save_model(save_path)

if smp.mp_rank() == 1:
    loss_np = train_loss.result().numpy()
    try:
        loss_scalar = loss_np.item()  # numpy >= 1.16
    except:
        loss_scalar = loss_np.asscalar()  # numpy < 1.16

    log_result("Training loss", loss_scalar)
    assert loss_scalar < 0.008


if smp.dp_rank() == 0 and smp.mp_rank() == 0:
    # testing saved model.
    assert os.path.exists(save_path) == True, save_path + " does not exist."
    loaded = tf.keras.models.load_model(save_path)
    infer = loaded.signatures["serving_default"]
    pred = None
    for images, labels in train_ds.take(1):
        pred = infer(images)
        res = pred["output_0"]

    result = np.argmax(res, axis=1)
    expected_result = labels.numpy()
    match = expected_result == result
    assert (
        match.tolist().count(True) / match.shape[0] > 0.9
    ), f"Accuracy less than 0.9 \n Expected: {expected_result} \n Got:      {result}"

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
