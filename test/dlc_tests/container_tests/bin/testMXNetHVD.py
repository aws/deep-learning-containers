import argparse, time, logging
import os

# Disable Autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.gluon.model_zoo.vision import get_model
import horovod.mxnet as hvd


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable training on GPU (default: False)')
    opt = parser.parse_args()
    return opt


def get_synthetic_dataloader(batch_size, num_workers):
    image_shape = (3, 256, 256)
    num_samples = 64
    num_classes = 1000
    data_shape = (num_samples,) + image_shape

    label = mx.nd.array(np.random.randint(0, num_classes, [num_samples, ]),
                        dtype=np.float32)
    data = mx.nd.array(np.random.uniform(-1, 1, data_shape),
                       dtype=np.float32)

    dataset = mx.gluon.data.dataset.ArrayDataset(data, label)
    train_data = mx.gluon.data.DataLoader(dataset, batch_size=batch_size,
                                          num_workers=num_workers, shuffle=True,
                                          last_batch='discard')
    return train_data


def train(net, train_data, batch_size, ctx, logging, args):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(), ctx=ctx)

    optimizer_params = {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum}
    optimizer = 'nag'
    hvd.broadcast_parameters(net.collect_params(), root_rank=0)
    trainer = hvd.DistributedTrainer(
        net.collect_params(),
        optimizer,
        optimizer_params
    )

    metric = mx.metric.Accuracy()
    train_metric = mx.metric.Accuracy()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    logging.info('Training Begins')

    for epoch in range(args.epochs):
        tic = time.time()
        train_metric.reset()
        metric.reset()
        train_loss = 0
        num_batch = len(train_data)

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            with ag.record():
                output = [net(X) for X in data]
                loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in loss])

            train_metric.update(label, output)

        train_loss /= batch_size * num_batch

        if hvd.rank() == 0:
            elapsed = time.time() - tic
            speed = num_batch * batch_size * hvd.size() / elapsed
            logging.info('Epoch[%d]\tSpeed=%.2f samples/s\tTime cost=%f',
                         epoch, speed, elapsed)


if __name__ == '__main__':
    args = parse_args()

    # Initialize Horovod
    hvd.init()

    if not args.no_cuda:
        # Disable CUDA if there are no GPUs.
        if not mx.test_utils.list_gpus():
            args.no_cuda = True

    # Horovod: pin context to local rank
    ctx = [mx.cpu(hvd.local_rank())] if args.no_cuda else [mx.gpu(hvd.local_rank())]
    per_worker_batch_size = args.batch_size // hvd.size()

    net = get_model('alexnet')
    net.hybridize()

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    # Training data
    train_data = get_synthetic_dataloader(per_worker_batch_size, args.num_workers)

    train(net, train_data, per_worker_batch_size, ctx, logging, args)

