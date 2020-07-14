import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

def test():
  img_path = tf.keras.utils.get_file('tensorflow.png','https://tensorflow.org/images/tf_logo.png')
  img_raw = tf.io.read_file(img_path)
  img = tf.io.decode_image(img_raw)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [500,500])

  # Simple test with tfa.image 
  mean = tfa.image.mean_filter2d(img, filter_shape=11)
  rotate = tfa.image.rotate(img, tf.constant(np.pi/8))
  transform = tfa.image.transform(img, [1.0, 1.0, -250, 0.0, 1.0, 0.0, 0.0, 0.0])

if __name__ == '__main__':
    test()