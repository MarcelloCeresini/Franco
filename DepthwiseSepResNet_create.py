import os
import pathlib
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu, sigmoid


print()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print()

# import the dataset
data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
  tf.keras.utils.get_file(
      'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)


# split the data 8:1:1
train_size = 0.8
val_size = 0.1

train_size = int(train_size * num_samples)
val_size = int(val_size * num_samples)
test_size = num_samples - train_size - val_size

train_files = filenames[:train_size]
val_files = filenames[train_size: train_size + val_size]
test_files = filenames[-val_size:]


# preprocess the data
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    # oss: we want ODD windows
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=32)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


def get_spectrogram_and_label_id(waveform, label):
    spectrogram = get_spectrogram(waveform)
    spectrogram = tf.expand_dims(spectrogram, -1)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id


AUTOTUNE = tf.data.AUTOTUNE


def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
    return output_ds


train_ds = preprocess_dataset(train_files)
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 128
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for element, _ in train_ds.take(1):
    input_shape = (element.shape[1], element.shape[2], element.shape[3])
print(input_shape)

# model definition
def DepthSeparableResidualBlock(x, filters):
    x_skip = x
    f1, f2 = filters

    # first block with downsampling
    x = layers.DepthwiseConv2D(kernel_size=3, strides=2, padding="same", kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)
    x = layers.Conv2D(f1, 1, 1, padding="valid", kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)

    # second block no downsampling
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)
    x = layers.Conv2D(f2, 1, 1, padding="valid", kernel_regularizer=l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    # no activation

    # we convolute the initial layer in order for it to have the same number of channels of the main branch
    x_skip = layers.Conv2D(f2, 1, 2, padding="valid", kernel_regularizer=l2(0.001))(x_skip)
    x_skip = layers.BatchNormalization()(x_skip)

    # residual addition and final activation
    x = layers.Add()([x, x_skip])
    x = layers.Activation(relu)(x)

    return x


def SqueezeExcitationLayer(x_init, ratio=16):
    channels = tf.keras.backend.int_shape(x_init)[-1]

    x = layers.GlobalAveragePooling2D()(x_init)
    x = layers.Dense(channels / ratio, activation="relu")(x)  # Bottleneck
    x = layers.Dense(channels, activation="sigmoid")(x)
    x = layers.Multiply()([x_init, x])

    return x


def KeywordRecognitionModel():

    inputs = layers.Input(input_shape)
    x = layers.Conv2D(64, 3, 2, padding="valid", kernel_regularizer=l2(0.001))(inputs)
    x = SqueezeExcitationLayer(x)
    x = DepthSeparableResidualBlock(x, (64, 128))
    outputs = DepthSeparableResidualBlock(x, (128, 256))

    return Model(inputs, outputs)

model = KeywordRecognitionModel()
model.summary()
