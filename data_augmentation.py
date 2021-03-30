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


def augment_dataset(ds, AUTOTUNE, preceding_timeframes = 5, batch_size=128):
    wd = preceding_timeframes + 1
    ds = ds.window(wd, drop_remainder=True)  # creates a dataset of datasets
    ds = ds.flat_map(lambda x: x.batch(wd))  # transforms the nested datasets into one big tensor, then it batches it into tensors of dimension "wd"
    ds = ds.map(lambda x: (AddNoise(x), x[-1:]), num_parallel_calls=AUTOTUNE)  # creates a "supervised style" dataset, input="#wd noisy frames", label="last clean frame"

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    # cache to speed up the training, look out for memory consumption (it will saturate the RAM for large datasets)
    # ds = ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return ds
