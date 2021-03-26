import tensorflow as tf

def get_spectrogram(waveform, label):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)

    # oss: we want ODD windows --> REALLY IMPORTANT PARAMS
    spectrogram = tf.signal.stft(
        equal_length,
        frame_length=255,
        frame_step=32)

    #  not clear if we want the pahse or not
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram, label


def preprocess_dataset(ds, AUTOTUNE, batch_size=128):
    output_ds = ds.map(get_spectrogram,  num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # cache to speed up the training, look out for memory consumption (it will saturate the RAM for large datasets)
    # output_ds = output_ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return output_ds
