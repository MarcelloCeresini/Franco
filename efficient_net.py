import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import softmax, relu
from tensorflow.nn import relu6

##############################################################
# A GOOD IDEA could be to DIFFERENCIATE between time and frequency
############################################################
weight_decay=1e-3
############################################################

def SqueezeExcitationLayer(x_init, ratio=4):  # if used IN the network, r=16 is preferred
    channels = tf.keras.backend.int_shape(x_init)[-1]

    x = layers.GlobalAveragePooling2D()(x_init)
    x = layers.Dense(int(channels / ratio))(x)  # Bottleneck
    x = layers.Activation(relu6)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(channels, activation="sigmoid")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Multiply()([x_init, x])

    return x


def InvertedBottleneckBlock(x, output_channels, name, expansion_ratio = 6, kernel_size=(3, 3), strides=(1, 1)):
    input_channels = tf.keras.backend.int_shape(x)[-1]
    x = layers.Conv2D(input_channels*expansion_ratio, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay), name=name+"PointConv1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu6)(x)
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding="same", kernel_regularizer=l2(weight_decay), name=name+"DepthConv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu6)(x)
    x = layers.Conv2D(output_channels, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay), name=name+"PointConv2")(x)
    x = layers.BatchNormalization()(x)
    # NO ACTIVATION --> the residual layer will be attached here and has to be left linear for better accuracy

    return x

# InvertedBottleneckResidualBlock
def IBRB(x, num_layers, channels, kernel_size, expansion_ratio, name):

    x_skip = x
    x_skip = layers.Conv2D(channels, kernel_size=(1,1), strides=(2, 2), padding="same", kernel_regularizer=l2(weight_decay), name=name+"ProjectionRes")(x_skip)
    x_skip = layers.BatchNormalization()(x_skip)
    x = InvertedBottleneckBlock(x, channels, name=name+"1_", expansion_ratio=expansion_ratio, kernel_size=(3, 3), strides=(2, 2))
    x = layers.Add()([x,x_skip])

    for i in range(2,num_layers):
        x_skip = x
        x = InvertedBottleneckBlock(x, channels, name=name+str(i)+"_", expansion_ratio=expansion_ratio, kernel_size=(3, 3), strides=(1, 1))
        x = layers.Add()([x, x_skip])

    return x


def EfficientNet(input_shape, num_classes, phi=1):
    ############################################################################
    alpha = 1.2
    beta = 1.1
    gamma = 1.15

    # d = alpha ** phi  # number of layers --> this has to be quantized:
    d = []
    d.append([2, 2, 3, 3, 4])  # total 14 --> phi = 1
    d.append([2, 3, 3, 4, 5])  # total 17 --> phi = 2
    d.append([2, 3, 4, 5, 6])  # total 20 --> phi = 3
    d.append([4, 4, 5, 5, 6])  # total 24 --> phi = 4
    w = beta ** phi   # number of channels inside a layer
    r = gamma ** phi  # resolution of the initial image
    ############################################################################

    inputs = layers.Input(input_shape)
    x = layers.Conv2D(int(32*w), 3, 2, padding="same", kernel_regularizer=l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu6)(x)
    x = SqueezeExcitationLayer(x)

    x = IBRB(x, num_layers=1,         channels=int(16*w), kernel_size=(3, 3), expansion_ratio=1, name="IBConv1_")
    x = IBRB(x, num_layers=d[phi-1][0], channels=int(24*w), kernel_size=(3, 3), expansion_ratio=6, name="IBConv6_b1_")
    x = IBRB(x, num_layers=d[phi-1][1], channels=int(40*w), kernel_size=(5, 5), expansion_ratio=6, name="IBConv6_b2_")
    x = IBRB(x, num_layers=d[phi-1][2], channels=int(80*w), kernel_size=(3, 3), expansion_ratio=6, name="IBConv6_b3_")
    x = IBRB(x, num_layers=d[phi-1][3], channels=int(112*w),kernel_size=(5, 5), expansion_ratio=6, name="IBConv6_b4_")
    x = IBRB(x, num_layers=d[phi-1][4], channels=int(192*w),kernel_size=(5, 5), expansion_ratio=6, name="IBConv6_b5_")
    x = IBRB(x, num_layers=1,         channels=int(320*w),kernel_size=(3, 3), expansion_ratio=6, name="IBConv6_b6_")

    x = layers.Conv2D(int(1280*w), kernel_size=(1, 1), strides=(1, 1), kernel_regularizer=l2(weight_decay), name="Conv2D_last")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(relu6)(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation(softmax)(x)

    return Model(inputs, outputs)
