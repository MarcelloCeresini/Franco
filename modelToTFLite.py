import os
import pathlib

print()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print()

import tensorflow as tf
import tensorflow_datasets as tfds
from data_preprocessing import get_spectrogram

data_dir = pathlib.Path('data/speech_commands')
ds_init = tfds.load(
    'speech_commands',
    split='train[:10%]',
    shuffle_files=True,
    data_dir=data_dir,
    as_supervised=True
)

AUTOTUNE = tf.data.AUTOTUNE


def representative_dataset():  # it needs an iterator
    for data in ds_init.map(get_spectrogram,  num_parallel_calls=AUTOTUNE).batch(1).take(100):
        yield [data.astype(tf.float32)]


converter = tf.lite.TFLiteConverter.from_saved_model("training/model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

os.system("edgetpu_compiler model.tflite")  # this is to make a call on the terminal
