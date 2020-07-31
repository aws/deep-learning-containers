import tensorflow as tf

dispatcher = tf.data.experimental.service.DispatchServer(port=50050)
dispatcher_address = dispatcher.target.split("://")[1]
worker = tf.data.experimental.service.WorkerServer(
    port=0, dispatcher_address=dispatcher_address)
print("Starting Worker")
worker.join()
