from invoke import run


def test_placeholder(mxnet_training):
    run("nvidia-smi")
