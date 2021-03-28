import tensorflow as tf
from tensorflow.python.ops import array_ops

# https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py

class FocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self,  alpha=0.25, gamma=2.0, from_logits=False, name="focal_crossentropy"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.squeeze(y_true)

        # ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        if self.from_logits:
            sigmoid_p = tf.sigmoid(y_pred)
        else:
            sigmoid_p = y_pred

        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        pos_p_sub = array_ops.where(y_true > zeros, y_true - sigmoid_p, zeros)
        neg_p_sub = array_ops.where(y_true > zeros, zeros, sigmoid_p)

        per_entry_cross_ent = - self.alpha * (pos_p_sub ** self.gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                              -(1-self.alpha)*(neg_p_sub**self.gamma)* tf.math.log(tf.clip_by_value(1.0-sigmoid_p,1e-8,1.0))

        return tf.reduce_sum(per_entry_cross_ent)
