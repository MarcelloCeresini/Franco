import os
import tensorflow as tf
import tensorflow_datasets as tfds
from data_preprocessing import preprocess_dataset

ds_init = tfds.load(
    'speech_commands',
    split='train[:1%]',
    shuffle_files=True,
    as_supervised=True
)

representative_dataset = preprocess_dataset(ds_init)

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
