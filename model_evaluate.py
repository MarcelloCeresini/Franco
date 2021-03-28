import os
import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model

from data_preprocessing import preprocess_dataset

#############################################
training_path = "training/model"

data_dir = pathlib.Path('data/speech_commands')
(train_ds, test_ds), ds_info = tfds.load("speech_commands",
    # split=["train[:1%]", "validation[:1%]", "test"],  # uncomment to test stuff
    split=["train", "test"],  # real training
    shuffle_files=True,
    data_dir=data_dir,
    as_supervised=True,
    with_info=True)

commands = ds_info.features["label"].names
batch_size = 1
AUTOTUNE = tf.data.AUTOTUNE
test_ds = preprocess_dataset(test_ds, AUTOTUNE, batch_size)

model = load_model(training_path)

# result = model.evaluate(test_ds)
# plot_history(training_path)
# confusion_matrix = conf_mtx(test_ds, commands)


def plot_history(training_path):

    with open(training_path + "_epoch_num.pkl", "rb") as f:
        num_epochs = pickle.load(f)
    with open(training_path + "_history.pkl", "rb") as f:
        history = pickle.load(f)

    init_epoch=1
    plt.plot(range(init_epoch,num_epochs+1), history["loss"][init_epoch-1:])
    plt.plot(range(init_epoch,num_epochs+1), history["val_loss"][init_epoch-1:])
    plt.legend(["loss", "val_loss"])
    plt.show()


def conf_mtx(test_ds, commands):
    confusion_matrix = np.zeros((len(commands), len(commands)))

    for audio, label in test_ds:
        # im = audio.numpy()
        # im = np.squeeze(im)
        # plt.imshow(im)
        # plt.show()
        y_true = label.numpy()[0]
        y_pred = np.argmax(model.predict(audio.numpy()))
        confusion_matrix[y_true, y_pred] += 1

    # plt.bar(range(0, len(commands)), y)
    # plt.show()

    plt.figure()
    sns.heatmap(confusion_matrix, xticklabels=commands, yticklabels=commands, annot=True, fmt="g")
    plt.xlabel("Prediction")
    plt.ylabel("True")
    plt.show()

    return confusion_matrix
