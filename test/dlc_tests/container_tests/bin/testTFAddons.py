import tensorflow as tf
import tensorflow_addons as tfa

def test():
  batch_size=64
  epochs=10
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(784,), activation='relu', name='dense_1'),
    tf.keras.layers.Dense(64, activation='relu', name='dense_2'),
    tf.keras.layers.Dense(10, activation='softmax', name='predictions'),
  ])
  
  # Load MNIST dataset as NumPy arrays
  dataset = {}
  num_validation = 10000
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  # Preprocess the data
  x_train = x_train.reshape(-1, 784).astype('float32') / 255
  x_test = x_test.reshape(-1, 784).astype('float32') / 255 

  # Compile the model
  model.compile(
      optimizer=tfa.optimizers.LazyAdam(0.001),  # Utilize TFA optimizer
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      metrics=['accuracy'])

  # Train the network
  history = model.fit(
      x_train,
      y_train,
      batch_size=batch_size,
      epochs=epochs)

  print('Evaluate on test data:')
  results = model.evaluate(x_test, y_test, batch_size=128, verbose = 2)
  print('Test loss = {0}, Test acc: {1}'.format(results[0], results[1]))


if __name__ == '__main__':
    test()