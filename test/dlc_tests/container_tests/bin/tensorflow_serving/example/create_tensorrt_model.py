## Resource Motivation: https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html

import os
import argparse
import tensorflow as tf


def run_training(model_save_folder_path=os.path.join("script_folder", "models")):
    # Define a simple sequential model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = tf.cast(x_train, dtype=tf.float32)
    y_train = tf.cast(y_train, dtype=tf.float32)
    x_test = tf.cast(x_test, dtype=tf.float32)
    y_test = tf.cast(y_test, dtype=tf.float32)
    # Train the model
    model.fit(x_train, y_train, epochs=1)

    # Evaluate your model accuracy
    model.evaluate(x_test, y_test, verbose=2)

    # Save model in the saved_model format
    SAVED_MODEL_DIR = os.path.join(model_save_folder_path, "native_saved_model")
    model.save(SAVED_MODEL_DIR)

    from tensorflow.python.compiler.tensorrt import trt_convert as trt

    # Instantiate the TF-TRT converter
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=SAVED_MODEL_DIR, precision_mode=trt.TrtPrecisionMode.FP32
    )

    # Convert the model into TRT compatible segments
    converter.convert()
    converter.summary()

    MAX_BATCH_SIZE = 128

    def input_fn():
        batch_size = MAX_BATCH_SIZE
        x = x_test[0:batch_size, :]
        yield [x]

    converter.build(input_fn=input_fn)

    OUTPUT_SAVED_MODEL_DIR = os.path.join(model_save_folder_path, "tftrt_saved_model")
    converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_save_folder_path",
        default=f"""{os.path.join("script_folder","models")}""",
        help="Location where builts model should be saved.",
    )
    args = parser.parse_args()
    print(1)
    run_training(model_save_folder_path=args.model_save_folder_path)


if __name__ == "__main__":
    main()
