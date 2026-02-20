import tensorflow as tf

dispatcher = tf.data.experimental.service.DispatchServer(
    tf.data.experimental.service.DispatcherConfig(port=50050), start=True
)
dispatcher_address = dispatcher.target.split("://")[1]
worker = tf.data.experimental.service.WorkerServer(
    tf.data.experimental.service.WorkerConfig(dispatcher_address=dispatcher_address), start=True
)
print("Starting Worker")
worker.join()
