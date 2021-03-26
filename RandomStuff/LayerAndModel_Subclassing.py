import tensorflow as tf
from tensorflow.keras import layers

class Linear(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        # or instead
        # self.w = self.add_weight(
        #     shape=(input_dim, units), initializer="random_normal", trainable=True
        # )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )
        # or instead
        # self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

        # NON trainable weights:
        # self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# call
linear_layer = Linear(units=16, input_dim=2)
print(linear_layer.weights) # or linear_layer.trainalbe_weights
# non trainable --> linear_layer.non_trainable_weights

'''
BEST RECCOMANDATION --> BUILD WEIGHTS LAZILY! Only after the size of the input is known, i.e. in the BUILD METHOD
build is called automatically the first time the layer is called
'''

class Linear(layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


'''
creating BLOCKS --> recursively compose Layers --> BEST PRACTICE: CALL THEM IN THE __init__()
'''


class Block(layers.Layer):
    def __init__(self):
        super(Block, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


'''
adding the LOSS method --> inside the call
oss --> losses are RESET at every __call__() to the TOP LEVEL LAYER --> so losses contain only values created during the 
forward pass
'''

class ActivityRegLayer(layers.Layer):
    def __init__(self, rate = 1e-2):
        super(ActivityRegLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate + tf.reduce_sum(inputs))
        return inputs


my_layer = ActivityRegLayer()
print(my_layer.losses)
# oss: also regularization losses created for the weights of the inner layers are stored in .losses

