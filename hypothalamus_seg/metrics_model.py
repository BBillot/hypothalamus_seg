# python imports
import tensorflow as tf
from keras.models import Model

# third-party imports
from ext.lab2im import layers


def metrics_model(input_model, metrics='dice'):

    # get prediction and gt
    last_tensor = input_model.outputs[0]
    labels_gt = input_model.get_layer('splitting').output[1]

    # make sure the tensors have the right keras shape
    last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
    labels_gt._keras_shape = tuple(labels_gt.get_shape().as_list())

    if metrics == 'dice':
        last_tensor = layers.DiceLoss()([labels_gt, last_tensor])

    elif metrics == 'wl2':
        last_tensor = layers.WeightedL2Loss(target_value=15)([labels_gt, last_tensor])

    else:
        raise Exception('metrics should either be "dice or "wl2, got {}'.format(metrics))

    # create the model and return
    model = Model(inputs=input_model.inputs, outputs=last_tensor)
    return model


class IdentityLoss(object):
    """Very simple loss, as the computation of the loss as been directly implemented in the model."""
    def __init__(self, keepdims=True):
        self.keepdims = keepdims

    def loss(self, y_true, y_predicted):
        """Because the metrics is already calculated in the model, we simply return y_predicted.
           We still need to put y_true in the inputs, as it's expected by keras."""
        loss = y_predicted

        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss
