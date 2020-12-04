# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model


def metrics_model(input_shape,
                  input_model=None,
                  metrics='dice',
                  name=None,
                  prefix=None):

    # naming the model
    model_name = name
    if prefix is None:
        prefix = model_name

    # first layer: input
    name = '%s_input' % prefix
    if input_model is None:
        input_tensor = KL.Input(shape=input_shape, name=name)
        last_tensor = input_tensor
    else:
        input_tensor = input_model.inputs
        last_tensor = input_model.outputs
        if isinstance(last_tensor, list):
            last_tensor = last_tensor[0]
        last_tensor = KL.Reshape(input_shape, name='predicted_output')(last_tensor)

    # get deformed labels
    n_labels = input_shape[-1]
    labels_gt = input_model.get_layer('splitting').output[1]
    labels_gt = KL.Reshape(input_shape)(labels_gt)
    labels_gt = KL.Lambda(lambda x: K.clip(x / K.sum(x, axis=-1, keepdims=True), K.epsilon(), 1))(labels_gt)

    # metrics is computed as part of the model
    if metrics == 'dice':

        # make sure predicted values are probabilistic
        last_tensor = KL.Lambda(lambda x: K.clip(x / K.sum(x, axis=-1, keepdims=True), K.epsilon(), 1))(last_tensor)

        # compute dice
        top = KL.Lambda(lambda x: 2*x[0]*x[1])([labels_gt, last_tensor])
        bottom = KL.Lambda(lambda x: K.square(x[0]) + K.square(x[1]))([labels_gt, last_tensor])
        for dims_to_sum in range(len(input_shape)-1):
            top = KL.Lambda(lambda x: K.sum(x, axis=1))(top)
            bottom = KL.Lambda(lambda x: K.sum(x, axis=1))(bottom)
        last_tensor = KL.Lambda(lambda x: x[0] / K.maximum(x[1], 0.001), name='dice')([top, bottom])  # 1d vector

        # compute mean dice loss
        w = np.ones([n_labels]) / n_labels
        last_tensor = KL.Lambda(lambda x: 1 - x, name='dice_loss')(last_tensor)
        last_tensor = KL.Lambda(lambda x: K.sum(x*tf.convert_to_tensor(w, dtype='float32'), axis=1),
                                name='mean_dice_loss')(last_tensor)
        # average mean dice loss over mini batch
        last_tensor = KL.Lambda(lambda x: K.mean(x), name='average_mean_dice_loss')(last_tensor)

    elif metrics == 'wl2':
        # compute weighted l2 loss
        weights = KL.Lambda(lambda x: K.expand_dims(1 - x[..., 0] + 1e-4))(labels_gt)
        normaliser = KL.Lambda(lambda x: K.sum(x[0]) * K.int_shape(x[1])[-1])([weights, last_tensor])
        last_tensor = KL.Lambda(
            lambda x: K.sum(x[2] * K.square(x[1] - (x[0] * 30 - 15))) / x[3],
            # lambda x: K.sum(x[2] * K.square(x[1] - (x[0] * 6 - 3))) / x[3],
            name='wl2')([labels_gt, last_tensor, weights, normaliser])

    else:
        raise Exception('metrics should either be "dice or "wl2, got {}'.format(metrics))

    # create the model and return
    model = Model(inputs=input_tensor, outputs=last_tensor, name=model_name)
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
