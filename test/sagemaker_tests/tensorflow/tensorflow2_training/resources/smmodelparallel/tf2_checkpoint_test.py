# Standard Library
import sys

# Third Party
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

# First Party
import smdistributed.modelparallel.tensorflow as smp

tf.random.set_seed(1234)

tf.config.optimizer.set_jit(len(sys.argv) > 1 and sys.argv[1] == "xla")

cfg = {
    "microbatches": 2,
    "placement_strategy": "spread",
    "partitions": 2,
    "contiguous": False,
    "xla": (len(sys.argv) > 1 and sys.argv[1] == "xla"),
    "pipeline": "interleaved",
    "optimize": "speed",
}
smp.init(cfg)


x_train = np.ones((8, 1))
y_train = np.ones((8, 1)) * 2


class MyModel(smp.DistributedModel):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(
            4, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=192)
        )
        self.d2 = Dense(1, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=126))

    def first(self, x):
        with tf.name_scope("first"):
            return self.d1(x)

    def second(self, x):
        with tf.name_scope("second"):
            return self.d2(x)

    def call(self, x, training=None):
        # x = self.conv0(x)
        x = self.first(x)
        return self.second(x)


# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.MSE


exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    0.085, decay_steps=5, decay_rate=0.96, staircase=False, name=None
)
optimizer = tf.keras.optimizers.Adam(learning_rate=exp_decay)


@smp.step
def get_grads(images, labels):
    # with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
    loss = tf.reduce_mean(loss)
    grads = optimizer.get_gradients(loss, model.trainable_variables)

    return grads, loss, predictions


@tf.function
def train_step(images, labels):
    gradients, loss, predictions = get_grads(images, labels)

    gradients = [g.accumulate() for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss.reduce_mean()


@smp.step(
    input_signature=[
        tf.TensorSpec(shape=[8, 1], dtype=tf.float32),
        tf.TensorSpec(shape=[8, 1], dtype=tf.float32),
    ]
)
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    t_loss = tf.reduce_mean(t_loss)

    return t_loss


test_variable = tf.Variable(0.0)

checkpoint_directory = "/tmp/tf2_ckpt_test/"

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, test_var=test_variable)

ckpt_manager = smp.CheckpointManager(checkpoint, checkpoint_directory)


def test_saved_ckpt():

    print("................................. Testing ckpt .......................................")

    ckpt_manager.restore(restore_prefix="ckpt-2")

    # One fwd step to get loss
    test_loss = test_step(x_train, y_train)
    test_loss = test_loss.reduce_mean()

    ckpt_obj = ckpt_manager.checkpoint

    assert (
        ckpt_obj.test_var.numpy() == 2.0
    ), f"Variable test_var not restored. \n Expected: 2.0 \n Got: {ckpt_obj.test_var.numpy()}"

    assert (
        test_loss == loss_list[2]
    ), f"Loss value did not match. \n Expected: {loss_list[-1]} \n Got: {test_loss}"


loss_list = []
# running 5 step with same inputs and labels
for step in range(5):

    if step == 1:
        ckpt_manager.restore(restore_prefix="ckpt-2")

    images = x_train
    labels = y_train

    res_loss = train_step(images, labels)
    loss_list.append(res_loss.numpy())
    test_variable.assign_add(1.0)
    print(f"Step: {step+1} loss: {res_loss.numpy()}")

    # Saving checkpoint
    ckpt_manager.save()

# Testing checkpoint.
test_saved_ckpt()

smp.barrier()


print("Test Complete")
