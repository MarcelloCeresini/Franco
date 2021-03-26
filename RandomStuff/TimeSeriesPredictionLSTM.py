import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import pandas as pd

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

import pandas as pd
import time

start = time.process_time()

data = pd.read_csv("/home/marcello/Downloads/ED.csv")

data = data.drop(columns="Date")
X_array = data.values.astype(np.float32)
y_array = data.Close.values.astype(np.float32)

num_samples = y_array.size
past_days = 64
batch_size = 256
buffer_size = 5000

X_train = []
y_train = []

for i in range(0, np.shape(data)[0] - past_days - 1):
    X_train.append(X_array[i:i + past_days])
    y_train.append(y_array[i + past_days])

X_train = tf.convert_to_tensor(X_train, dtype=np.float32)
X_train_CRNN = tf.expand_dims(X_train, -1)
y_train = tf.convert_to_tensor(y_train, dtype=np.float32)

ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds_CRNN = tf.data.Dataset.from_tensor_slices((X_train_CRNN, y_train))


def train_val_test_split(ds, train_size=0.8, val_size=0.1):

    dataset = ds.shuffle(buffer_size)

    train_size = int(train_size * num_samples)
    val_size = int(val_size * num_samples)
    test_size = num_samples - train_size - val_size

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size).take(test_size)

    train_ds = train_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)

    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = train_val_test_split(ds)
train_ds_CRNN, val_ds_CRNN, test_ds_CRNN = train_val_test_split(ds_CRNN)

a = 0
for element in ds.take(1):
    a = element[0].shape
    print(a)
input_shape = (a[0], a[1])

norm_layer = preprocessing.Normalization()
norm_layer.adapt(train_ds.map(lambda x, _: x))
#
# model = models.Sequential([
#     layers.Input(input_shape),
#     norm_layer,
#     layers.LSTM(512, activation="relu"),
#     layers.Dropout(0.05),
#     layers.Dense(1)
# ])
#
# model.summary()
LAST_CONV_DIM = 128
RESHAPE_DIM = int(((input_shape[0] - 4) / 2))

model_CRNN = models.Sequential([
    layers.Input((input_shape[0], input_shape[1], 1)),
    norm_layer,
    layers.Conv2D(256, 3, activation="relu"),
    layers.Conv2D(LAST_CONV_DIM, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Reshape((RESHAPE_DIM, LAST_CONV_DIM)),
    layers.Dropout(0.05),
    layers.LSTM(256, activation="relu"),
    layers.Dropout(0.05),
    layers.Dense(1)
])

model_CRNN.summary()

# model.compile(
#     optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0005),
#     loss=tf.keras.losses.MeanSquaredError(),
#     metrics=tf.keras.metrics.MeanSquaredError(),
# )

model_CRNN.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=tf.keras.metrics.MeanSquaredError(),
)

EPOCHS = 100

# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     batch_size=batch_size,
#     validation_batch_size=batch_size,
#     epochs=1,
#     callbacks=tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
# )

history_CRNN = model_CRNN.fit(
    train_ds_CRNN,
    validation_data=val_ds_CRNN,
    batch_size=batch_size,
    validation_batch_size=batch_size,
    epochs=EPOCHS,
    # callbacks=tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
)

print(time.process_time() - start)

ds = ds.batch(batch_size)
# y_pred = model.predict(ds.map(lambda x, _: x))
ds_CRNN = ds_CRNN.batch(batch_size)
y_pred_CRNN = model_CRNN.predict(ds_CRNN.map(lambda x, _: x))

print("y_train: mean", y_train.numpy().mean())
# print("y_pred: mean", y_pred.mean())
print("y_pred_CRNN: mean", y_pred_CRNN.mean())
print("y_train: std", y_train.numpy().std())
# print("y_pred: std", y_pred.std())
print("y_pred_CRNN: std", y_pred_CRNN.std())

val = 0
plot1 = plt.figure(1)
# plt.plot(history.history["loss"][val:])
# plt.plot(history.history["val_loss"][val:])
plt.plot(history_CRNN.history["loss"][val:])
plt.plot(history_CRNN.history["val_loss"][val:])

plot2 = plt.figure(2)
plt.plot(y_train.numpy())
# plt.plot(y_pred)
plt.plot(y_pred_CRNN)

# result = model.evaluate(test_ds)
result_CRNN = model_CRNN.evaluate(test_ds_CRNN)
# print(result)
print(result_CRNN)
