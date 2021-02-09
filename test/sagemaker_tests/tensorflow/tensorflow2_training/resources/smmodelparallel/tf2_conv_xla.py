# Standard Library
import errno
import os
import shutil
import sys
import time

# Third Party
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from utils import log_result

# First Party
import smdistributed.modelparallel.tensorflow as smp

tf.random.set_seed(1234)

tf.config.optimizer.set_jit(True)

cfg = {
    "microbatches": 2,
    "partitions": 2,
    "placement_strategy": "spread",
    "pipeline": "interleaved",
    "optimize": "memory",
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

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000, seed=123).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


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
        self.last = Dense(10, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=129))

    def first(self, x):
        with tf.name_scope("first"):
            x = self.conv1(x)
            return self.flatten(x)

    def second(self, x):
        with tf.name_scope("second"):
            x = self.d1(x)
            x = self.d2(x)

            # testing layer re-use
            x = self.last(x)
            return self.last(x)

    def call(self, x, training=None):
        # x = self.conv0(x)
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
test_loss = tf.keras.metrics.Mean(name="test_loss")
train_loss = tf.keras.metrics.Mean(name="train_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")


@smp.step
def get_grads(images, labels):
    # with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)

    grads = optimizer.get_gradients(loss, model.trainable_variables)

    return grads, loss, predictions


@tf.function
def train_step(images, labels):
    gradients, loss, predictions = get_grads(images, labels)

    gradients = [g.accumulate() for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss.reduce_mean())
    train_accuracy(labels, predictions.merge())
    return loss.reduce_mean()


@smp.step(
    input_signature=[
        tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float64),
        tf.TensorSpec(shape=[None], dtype=tf.uint8),
    ]
)
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)
    return t_loss


checkpoint_directory = "/tmp/tf2_conv_checkpoints"


checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

ckpt_manager = smp.CheckpointManager(checkpoint, checkpoint_directory)


duration = 0.0
for epoch in range(5):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    t0 = None
    step = 0

    for images, labels in train_ds:
        if step == 1:
            t0 = time.time()
            if epoch == 0:
                ckpt_manager.restore()

        train_step(images, labels)

        step += 1
    duration += time.time() - t0

    ckpt_manager.save()

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        "Epoch {}, Loss: {}, Accuracy: {}, Test loss {}, test accuracy {}".format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result(),
            test_loss.result(),
            test_accuracy.result(),
        )
    )

t1 = time.time()
xla_str = " (XLA)" if tf.config.optimizer.get_jit() else ""
log_result("Time-to-train" + xla_str, duration)
assert t1 - t0 < 72.0


save_path = "./tf2_conv_saved_model"
model.save_model(save_path)

if smp.rank() == 0:
    loss_np = train_loss.result().numpy()
    try:
        loss_scalar = loss_np.item()  # numpy >= 1.16
    except:
        loss_scalar = loss_np.asscalar()  # numpy < 1.16

    log_result("Training loss" + xla_str, loss_scalar)
    assert loss_scalar < 0.016

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

    # Test saved checkpoints
    assert os.path.exists(checkpoint_directory) == True, f"Checkpoint directory  was not created"
    assert (
        os.path.exists(os.path.join(checkpoint_directory, "mp_rank_0")) == True
    ), f" Rank 0 directory does not exist"
    assert (
        os.path.exists(os.path.join(checkpoint_directory, "mp_rank_1")) == True
    ), f" Rank 1 directory does not exist"
    assert os.path.exists(
        os.path.join(checkpoint_directory, "mp_rank_0", "checkpoint")
    ), f" No checkpoint file found"
    assert os.path.exists(
        os.path.join(checkpoint_directory, "mp_rank_1", "checkpoint")
    ), f" No checkpoint file found"

    if os.path.exists(checkpoint_directory):
        shutil.rmtree(checkpoint_directory)
