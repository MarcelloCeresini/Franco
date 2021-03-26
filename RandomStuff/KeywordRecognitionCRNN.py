import os
import pathlib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display



# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

data_dir = pathlib.Path('data/mini_speech_commands')
if not data_dir.exists():
      tf.keras.utils.get_file(
          'mini_speech_commands.zip',
          origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
          extract=True,
          cache_dir='.', cache_subdir='data')

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
# print('Number of total examples:', num_samples)
# print('Number of examples per label:',
#       len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
# print('Example file tensor:', filenames[0])

train_size = 0.8
val_size = 0.1

train_size = int(train_size * num_samples)
val_size = int(val_size * num_samples)
test_size = num_samples - train_size - val_size

train_files = filenames[:train_size]
val_files = filenames[train_size: train_size + val_size]
test_files = filenames[-val_size:]


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
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=64)

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

T = input_shape[0]
F = input_shape[1]
W = 11  # to set as an odd number
# (total frame length is even, window for stft has to be odd, stride is even, so input_shape is odd) -->
# to make the convolution divisible by 2 (stride = 2) we need W odd
L = 11
St = 2
Sf = 2
CONV_NEURONS = 75
GRU_NEURONS1 = 50
GRU_NEURONS2 = 50
DENSE_NEURONS = 150

inputs = layers.Input(input_shape)
x = layers.Conv2D(CONV_NEURONS, kernel_size=(W, L), strides=(St, Sf), activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Reshape((int((T - W) / St + 1), int((F - L) / Sf + 1) * CONV_NEURONS))(x)
x = layers.Bidirectional(layers.GRU(GRU_NEURONS1, return_sequences=True))(x)
x = layers.LayerNormalization()(x)
x = layers.Bidirectional(layers.GRU(GRU_NEURONS2))(x)
x = layers.LayerNormalization()(x)
x = layers.Dense(DENSE_NEURONS)(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(10)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics="accuracy",
)

checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
save_frequency = 1
# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))
'''
# used to retrieve the checkpoints
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
'''
EPOCHS = 100
history = model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=batch_size,
    validation_batch_size=batch_size,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_weights_only=True,
                                           verbose=1,
                                           save_freq=save_frequency*batch_size)
    ]
)

result = model.evaluate(test_ds)

val = 0
plot1 = plt.figure(1)
plt.plot(history.history["loss"][val:])
plt.plot(history.history["val_loss"][val:])

os.mkdir("saved_model")
model.save('saved_model/my_model')
'''
# used to retrieve whole model (in case 100 EPOCHS are not enough, you can resume training from this model)
new_model = tf.keras.models.load_model('saved_model/my_model')
'''

converter = tf.lite.TFLiteConverter.from_saved_model("saved_model/my_model")
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
