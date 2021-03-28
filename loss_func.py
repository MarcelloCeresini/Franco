import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

'''
################################################################################
class LossFunctionWrapper(tf.keras.losses.Loss):
    def __init__(self,
               fn,
               reduction=losses_utils.ReductionV2.AUTO,
               name=None,
               **kwargs):
        super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs
    def call(self, y_true, y_pred):
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
            y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(
                y_pred, y_true)
        ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
        return ag_fn(y_true, y_pred, **self._fn_kwargs)
    def get_config(self):
        config = {}
        for k, v in six.iteritems(self._fn_kwargs):
            config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
################################################################################
class FocalCrossEntropy(LossFunctionWrapper):

    def __init__(self,
    	alpha=0.25,
    	gamma=2.0,
    	reduction=losses_utils.ReductionV2.AUTO,
    	name='focal_crossentropy'):
        super(FocalCrossEntropy, self).__init__(
            focal_crossentropy, name=name, reduction=reduction,
            alpha=alpha, gamma=gamma)
################################################################################
'''
def focal_crossentropy_func(alpha=0.25, gamma=2.0, from_logits=False):
    def focal_crossentropy(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        t_true = tf.squeeze(y_true)

        if from_logits:
            sigmoid_p = tf.sigmoid(y_pred)
        else:
            sigmoid_p = y_pred

        zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        pos_p_sub = tf.where(y_true > zeros, y_true - sigmoid_p, zeros)
        neg_p_sub = tf.where(y_true > zeros, zeros, sigmoid_p)

        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                                  -(1-alpha)*(neg_p_sub**gamma)* tf.math.log(tf.clip_by_value(1.0-sigmoid_p,1e-8,1.0))

        return tf.reduce_sum(per_entry_cross_ent)
    return focal_crossentropy
