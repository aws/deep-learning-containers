class _Stub:
    def __getattr__(self, name):
        raise RuntimeError(
            "mlio is not available in this image. "
            "SageMaker Pipe mode and RecordIO Protobuf format "
            "are not supported. Use File mode with CSV, Parquet, "
            "or LibSVM format instead."
        )


_stub = _Stub()


def __getattr__(name):
    return _stub
