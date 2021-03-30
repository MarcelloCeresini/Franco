import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu, sigmoid

##############################################################
# A GOOD IDEA could be to DIFFERENCIATE between time and frequency
############################################################
weight_decay=1e-3
############################################################

# model definition
def DepthSeparableResidualBlock(x, filters, name, strides=(1, 1)):
    x_skip = x
    f1, f2 = filters

    # first block with downsampling
    x = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding="same", kernel_regularizer=l2(weight_decay), name=name+"DepthConv1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)
    x = layers.Conv2D(f1, 1, 1, padding="valid", kernel_regularizer=l2(weight_decay), name=name+"PointConv1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)

    # second block no downsampling
    x = layers.DepthwiseConv2D(kernel_size=3, strides=1, dilation_rate=(2, 1), padding="same", kernel_regularizer=l2(weight_decay), name=name+"DepthConv2")(x)
    # x = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="same", kernel_regularizer=l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)
    x = layers.Conv2D(f2, 1, 1, padding="valid", kernel_regularizer=l2(weight_decay), name=name+"PointConv2")(x)
    x = layers.BatchNormalization()(x)
    # no activation

    if strides!=(1, 1):
        x_skip = layers.Conv2D(f2, 1, strides=strides, padding="valid", kernel_regularizer=l2(weight_decay), name=name+"ProjectionRes")(x_skip)
        x_skip = layers.BatchNormalization()(x_skip)

    # residual addition and final activation
    x = layers.Add()([x, x_skip])
    x = layers.Activation(relu)(x)

    return x


def SqueezeExcitationLayer(x_init, ratio=16):
    channels = tf.keras.backend.int_shape(x_init)[-1]

    x = layers.GlobalAveragePooling2D()(x_init)
    x = layers.Dense(channels / ratio, activation="relu")(x)  # Bottleneck
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(channels, activation="sigmoid")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Multiply()([x_init, x])

    return x


def KeywordRecognitionModel(input_shape, num_classes):

    inputs = layers.Input(input_shape)
    x = layers.Conv2D(64, 7, 2, padding="valid", kernel_regularizer=l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = SqueezeExcitationLayer(x)

    x = DepthSeparableResidualBlock(x, (64, 256), strides=(2, 2), name="conv2_1_")
    for i in range(2):
        x = DepthSeparableResidualBlock(x, (64, 256), name="conv2_" + str(i+2) + "_")

    x = DepthSeparableResidualBlock(x, (128, 512), strides=(2, 2), name="conv3_1_")
    for i in range(3):
        x = DepthSeparableResidualBlock(x, (128, 512), name="conv3_" + str(i+2) + "_")

    x = DepthSeparableResidualBlock(x, (256, 1024), strides=(2, 2), name="conv4_1_")
    for i in range(5):
        x = DepthSeparableResidualBlock(x, (256, 1024), name="conv4_" + str(i+2) + "_")

    x = DepthSeparableResidualBlock(x, (512, 2048), strides=(2, 2), name="conv5_1_")
    for i in range(2):
        x = DepthSeparableResidualBlock(x, (512, 2048), name="conv5_" + str(i+2) + "_")

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)
