import argparse
from mxnet import gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
import time
import mxnet as mx
from smdebug.mxnet import Hook, SaveConfig, modes
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a mxnet gluon model for FashonMNIST dataset"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of Epochs")
    parser.add_argument(
        "--smdebug_path",
        type=str,
        default=None,
        help="S3 URI of the bucket where tensor data will be stored.",
    )
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--random_seed", type=bool, default=False)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Reduce the number of training "
        "and evaluation steps to the give number if desired."
        "If this is not passed, trains for one epoch "
        "of training and validation data",
    )
    parser.add_argument(
        "--context",
        type=str,
        default='cpu',
        help="Context can be either cpu or gpu",
    )

    opt = parser.parse_args()
    return opt


def acc(output, label):
    return (output.argmax(axis=1) == label.astype("float32")).mean().asscalar()


def train_model(batch_size, net, ctx, train_data, valid_data, lr, hook, num_epochs, num_steps=None):
    net.initialize(init=init.Xavier(), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    hook.register_hook(softmax_cross_entropy)
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": lr})
    # Start the training.
    for epoch in range(num_epochs):
        train_loss, train_acc, valid_acc = 0.0, 0.0, 0.0
        tic = time.time()
        hook.set_mode(modes.TRAIN)
        for i, (data, label) in enumerate(train_data):
            if num_steps is not None and num_steps < i:
                break
            data = data.as_in_context(ctx)
            # forward + backward
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            # update parameters
            trainer.step(batch_size)
            # calculate training metrics
            train_loss += loss.mean().asscalar()
            train_acc += acc(output, label)
        # calculate validation accuracy
        hook.set_mode(modes.EVAL)
        for i, (data, label) in enumerate(valid_data):
            if num_steps is not None and num_steps < i:
                break
            data = data.as_in_context(ctx)
            valid_acc += acc(net(data), label)
        print(
            "Epoch %d: loss %.3f, train acc %.3f, test acc %.3f, in %.1f sec"
            % (
                epoch,
                train_loss / len(train_data),
                train_acc / len(train_data),
                valid_acc / len(valid_data),
                time.time() - tic,
            )
        )


def prepare_data(batch_size):
    mnist_train = datasets.FashionMNIST(train=True)
    X, y = mnist_train[0]
    ("X shape: ", X.shape, "X dtype", X.dtype, "y:", y)
    text_labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    X, y = mnist_train[0:10]
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.13, 0.31)])
    mnist_train = mnist_train.transform_first(transformer)
    train_data = gluon.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    mnist_valid = gluon.data.vision.FashionMNIST(train=False)
    valid_data = gluon.data.DataLoader(
        mnist_valid.transform_first(transformer), batch_size=batch_size, num_workers=4
    )
    return train_data, valid_data


# Create a model using gluon API. The smdebug hook is currently
# supports MXNet gluon models only.
def create_gluon_model():
    # Create Model in Gluon
    net = nn.HybridSequential()
    net.add(
        nn.Conv2D(channels=6, kernel_size=5, activation="relu"),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation="relu"),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(10),
    )
    return net


# Create a smdebug hook. The initialization of hook determines which tensors
# are logged while training is in progress.
# Following function shows the default initialization that enables logging of
# weights, biases and gradients in the model.
def create_smdebug_hook(output_s3_uri):
    # With the following SaveConfig, we will save tensors for steps 1, 2 and 3
    # (indexing starts with 0).
    save_config = SaveConfig(save_steps=[1, 2, 3])

    # Create a hook that logs weights, biases and gradients while training the model.
    hook = Hook(
        out_dir=output_s3_uri,
        save_config=save_config,
        include_collections=["weights", "gradients"],
    )
    return hook


def main():
    opt = parse_args()

    # these random seeds are only intended for test purpose.
    # for now, 128,12,2 could promise no assert failure with running smdebug_rules test_rules.py and config.yaml
    # if you wish to change the number, notice that certain steps' tensor value may be capable of variation
    if opt.random_seed:
        mx.random.seed(128)
        random.seed(12)
        np.random.seed(2)

    # Create a Gluon Model.
    net = create_gluon_model()

    # Create a smdebug hook for logging the desired tensors.
    out_dir = opt.smdebug_path

    hook = create_smdebug_hook(out_dir)



    # Register the hook to the top block.
    hook.register_hook(net)

    # Start the training.
    batch_size = opt.batch_size
    train_data, valid_data = prepare_data(batch_size)

    context = mx.cpu() if opt.context.lower() == 'cpu' else mx.gpu()
    # Start the training.
    train_model(batch_size, net, context, train_data, valid_data, opt.learning_rate, hook, opt.epochs, opt.num_steps)


    print ("Training is complete")


    from smdebug.trials import create_trial
    print("Created the trial with out_dir {0}".format(out_dir))
    tr = create_trial(out_dir)
    assert tr
    print ("Train steps: "  + str(tr.steps(mode=modes.TRAIN)))
    print ("Eval steps: " + str(tr.steps(mode=modes.EVAL)))
    assert len(tr.steps(mode=modes.TRAIN)) == 4
    assert len(tr.steps(mode=modes.EVAL)) == 4

    tnames = tr.tensor_names(regex="^conv._weight")
    print(tnames)
    assert (len(tnames) == 2)
    tname = tnames[0]

    # The tensor values will be available for 8 steps (4 TRAIN anf 4 EVAL)
    assert(len(tr.tensor(tname).steps()) == 8)

    loss_tensor_name = tr.tensor_names(regex="softmaxcrossentropyloss._output_.")[0]
    print("Obtained the loss tensor " + loss_tensor_name)

    #We have only one loss block
    assert (loss_tensor_name == 'softmaxcrossentropyloss0_output_0')

    # Number of elements in loss tensor should match the batch size.
    loss_tensor_value = tr.tensor(loss_tensor_name).value(step_num=2)
    assert len(loss_tensor_value) == batch_size

    mean_loss_tensor_value = tr.tensor(loss_tensor_name).reduction_value(step_num=2, reduction_name="mean", abs=False)
    print("Mean validation loss = " + str(mean_loss_tensor_value))

    print("Validation Complete")



if __name__ == "__main__":
    main()
