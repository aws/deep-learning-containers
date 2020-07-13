import tensorflow as tf
import tensorflow_addons as tfa

def train():
  mnist = tf.keras.datasets.mnist

  (x_train, y_train),(x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  model = tf.keras.models.Sequential([
    # Reshape into "channels last" setup.
    tf.keras.layers.Reshape((28,28,1), input_shape=(28,28)),
    tf.keras.layers.Conv2D(filters=10, kernel_size=(3,3),data_format="channels_last"),
    # LayerNorm Layer
    tfa.layers.InstanceNormalization(axis=3, 
                                     center=True, 
                                     scale=True,
                                     beta_initializer="random_uniform",
                                     gamma_initializer="random_uniform"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(x_test, y_test)

  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

if __name__ == '__main__':
    train()