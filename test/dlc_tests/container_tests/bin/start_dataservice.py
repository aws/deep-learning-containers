import tensorflow as tf

dispatcher = tf.data.experimental.service.DispatchServer(tf.data.experimental.service.DispatcherConfig(port=50050))
dispatcher_address = dispatcher.target.split("://")[1]
worker = tf.data.experimental.service.WorkerServer(tf.data.experimental.service.WorkerConfig(
         dispatcher_address=dispatcher_address))
print("Starting Worker")
worker.join()
