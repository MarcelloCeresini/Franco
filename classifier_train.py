import os
import pathlib
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.append("~/github/Ramassin/resumable_model")

from resumable_model.models import ResumableModel

import tensorflow as tf
import tensorflow_datasets as tfds
from loss_func import focal_crossentropy_func
# from focal_loss import SparseCategoricalFocalLoss

from data_preprocessing import preprocess_dataset
# from classifier_definition import KeywordRecognitionModel
from efficient_net import EfficientNet

print()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print()

# import data
data_dir = pathlib.Path('data/speech_commands')
(train_ds, val_ds), ds_info = tfds.load("speech_commands",
    # split=["train[:10%]", "validation[:10%]"],  # uncomment to test stuff
    split=["train", "validation"],  # real training
    shuffle_files=True,
    data_dir=data_dir,
    as_supervised=True,
    with_info=True)
num_classes = ds_info.features["label"].num_classes
################################################################################
batch_size = 16
AUTOTUNE = tf.data.AUTOTUNE
train_ds = preprocess_dataset(train_ds, AUTOTUNE, batch_size, one_hot = num_classes)
val_ds = preprocess_dataset(val_ds, AUTOTUNE, batch_size, one_hot = num_classes)
################################################################################

for element, label in train_ds.take(1):
    input_shape = (element.shape[1], element.shape[2], element.shape[3])

model = EfficientNet(input_shape, num_classes, phi=2)

optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)
loss = focal_crossentropy_func(alpha=0.25, gamma=2.0)
metric = tf.keras.metrics.AUC(num_thresholds=200, curve='PR')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metric
)

# training_path = "training/model"  # this for the KeywordRecognitionModel
training_path = "training_efficient/model"  # this for efficient_net
training_dir = os.path.dirname(training_path)
if not os.path.exists(training_dir):
    os.mkdir(training_dir)

custom_objects = { "focal_crossentropy": loss }
# with tf.keras.utils.custom_object_scope(custom_objects):
resumable_model = ResumableModel(model, to_path=training_path, custom_objects=custom_objects)

TOTAL_EPOCHS = 26
history = resumable_model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=batch_size,
    validation_batch_size=batch_size,
    epochs=TOTAL_EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = 10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor = 0.1, patience = 5, verbose=1)
    ]
)

# model.save(training_path)
