
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu


############################################################
weight_decay=1e-3
############################################################

def CalculateCrop(dim_x, dim_x_res):
    x_diff = dim_x[0] - dim_x_res[0]
    y_diff = dim_x[1] - dim_x_res[1]

    if x_diff % 2 == 0:
        crop_x = (int(x_diff / 2), int(x_diff / 2))
    else:
        crop_x = (int(x_diff / 2 + 1), int(x_diff / 2))

    if y_diff % 2 == 0:
        crop_y = (int(y_diff / 2), int(y_diff / 2))
    else:
        crop_y = (int(y_diff / 2 + 1), int(y_diff / 2))

    return crop_x, crop_y

# model definition
def DepthSeparableBlock(x, filters, name, activation=True):

    # first block with downsampling
    x = layers.DepthwiseConv2D(kernel_size=3, padding="same", kernel_regularizer=l2(weight_decay), name=name+"DepthConv1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu)(x)
    x = layers.Conv2D(filters, 1, padding="same", kernel_regularizer=l2(weight_decay), name=name+"PointConv1")(x)
    x = layers.BatchNormalization()(x)

    if activation:  # needed for the residual connections
        x = layers.Activation(relu)(x)

    return x


# def SqueezeExcitationLayer(x_init, ratio=16):
#     channels = tf.keras.backend.int_shape(x_init)[-1]
#
#     x = layers.GlobalAveragePooling2D()(x_init)
#     x = layers.Dense(channels / ratio, activation="relu")(x)  # Bottleneck
#     x = layers.Dropout(0.5)(x)
#     x = layers.Dense(channels, activation="sigmoid")(x)
#     x = layers.Dropout(0.5)(x)
#     x = layers.Multiply()([x_init, x])
#
#     return x


def DenoisingAutoencoderModel(input_shape):

    filters = [128, 256, 512]
    encoding_layers = 3

    inputs = layers.Input(input_shape)
    x = inputs
    x_res = [inputs]

    # Encoding part
    for i in range(encoding_layers):
        x = DepthSeparableBlock(x, filters[i], name="conv"+str(i)+"a_")
        x = layers.MaxPool2D(pool_size=(2, 2), padding="same")(x)

        x = DepthSeparableBlock(x, filters[i], name="conv"+str(i)+"b_")
        x = layers.MaxPool2D(pool_size=(2, 2), padding="same")(x)
        x_res.append(x)

    # Encoded Representation
    x = DepthSeparableBlock(x, 1024, name="conv5_", activation=False)

    # Decoding Part
    for i in range(encoding_layers):
        j = encoding_layers - i

        x = DepthSeparableBlock(x, filters[j-1], name="conv"+str(i+encoding_layers+2)+"a_", activation=False)

        # Residual Addition
        crop = CalculateCrop(tf.keras.backend.int_shape(x)[1:3], tf.keras.backend.int_shape(x_res[j])[1:3])
        x = layers.Cropping2D(crop)(x)
        x = layers.Add()([x, x_res[j]])
        x = layers.Activation(relu)(x)

        x = layers.UpSampling2D(size=(2, 2))(x)

        x = DepthSeparableBlock(x, filters[j-1], name="conv"+str(i+encoding_layers+2)+"b_")
        x = layers.UpSampling2D(size=(2, 2))(x)

    print(tf.keras.backend.int_shape(x))
    x = layers.Conv2D(input_shape[-1], kernel_size=(1, 1), name="PointWise_conv_LastResidual", kernel_regularizer=l2(weight_decay))(x)
    crop = CalculateCrop(tf.keras.backend.int_shape(x)[1:3], tf.keras.backend.int_shape(x_res[0])[1:3])
    x = layers.Cropping2D(crop)(x)
    x = layers.Add()([x, x_res[0]])
    x = layers.Activation(relu)(x)

    outputs = layers.Conv2D(1, kernel_size=(1, 1), name="Final_PoinWise_convolution", kernel_regularizer=l2(weight_decay))(x)

    return Model(inputs, outputs)
