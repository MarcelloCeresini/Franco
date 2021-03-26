import os
import pathlib
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow_datasets as tfds

from data_preprocessing import preprocess_dataset
from model_definition import KeywordRecognitionModel

print()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print()

# import data
data_dir = pathlib.Path('data/speech_commands')
(train_ds, val_ds, test_ds), ds_info = tfds.load("speech_commands",
    split=["train", "validation", "test"],
    shuffle_files=True,
    data_dir=data_dir,
    as_supervised=True,
    with_info=True)

################################################################################
batch_size = 16
AUTOTUNE = tf.data.AUTOTUNE
train_ds = preprocess_dataset(train_ds, AUTOTUNE, batch_size)
val_ds = preprocess_dataset(val_ds, AUTOTUNE, batch_size)
################################################################################

for element, label in train_ds.take(1):
    input_shape = (element.shape[1], element.shape[2], element.shape[3])
num_classes = ds_info.features["label"].num_classes

model =  KeywordRecognitionModel(input_shape, num_classes)

checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

check = pathlib.Path(checkpoint_dir)
if not check.exists():  # --> initialize model
    # model.summary()
    model.save_weights(checkpoint_path.format(epoch=0))
else:
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics="accuracy"
)

print("Start training")
EPOCHS = 1
history = model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=batch_size,
    validation_batch_size=batch_size,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(factor = 0.1, patience = 5, verbose=1)
    ]
)

# model.save('saved_model/my_model_finished')
