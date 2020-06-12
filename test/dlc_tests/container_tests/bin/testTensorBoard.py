import numpy as np
import tensorflow as tf
from tensorflow import keras
def train():
   print('TF version: {}'.format(tf.version))
   model = keras.Sequential(
       keras.layers.Flatten(input_shape=(28, 28)),
       keras.layers.Dense(128, activation='relu'),
       keras.layers.Dense(10)
   )
   model.compile(optimizer='adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics='accuracy')
   callbacks = tf.keras.callbacks.TensorBoard(log_dir='logs', profile_batch=1)
   X_train = np.random.rand(1,28,28)
   Y_train = np.random.rand(1,)
   model.fit(X_train, Y_train,  epochs=0, callbacks=callbacks)

if __name__ == '__main__':
   train()
