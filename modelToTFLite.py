import os
import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train_init, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def representative_dataset():  # to QUANTIZE the result and the weights
    for data in ds_train_init.map(normalize_img).batch(1).take(int(ds_info.splits['train'].num_examples/10)):
        yield [data[0]]


converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

os.system("edgetpu_compiler model.tflite")  # this is to make a call on the terminal
