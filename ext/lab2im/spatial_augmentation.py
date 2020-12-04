# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K

# project imports
from . import utils
from . import edit_volumes

# third-party imports
import ext.neuron.layers as nrn_layers


def deform_tensor(tensor,
                  affine_trans=None,
                  apply_elastic_trans=True,
                  inter_method='linear',
                  nonlin_std=2.,
                  nonlin_shape_factor=.0625,
                  additional_tensor=None,
                  additional_inter_method='linear'):
    """This function spatially deforms a tensor with a combination of affine and elastic transformations.
    :param tensor: input tensor to deform. Expected to have shape [batchsize, shape_dim1, ..., shape_dimn, channel].
    :param affine_trans: (optional) tensor of shape [batchsize, n_dims+1, n_dims+1] corresponding to an affine 
    transformation. Default is None, no affine transformation is applied.
    :param apply_elastic_trans: (optional) whether to deform the input tensor with a diffeomorphic elastic 
    transformation. If True the following steps occur:
    1) a small-size SVF is sampled from a centred normal distribution of random standard deviation.
    2) it is resized with trilinear interpolation to half the shape of the input tensor
    3) it is integrated to obtain a diffeomorphic transformation
    4) finally, it is resized (again with trilinear interpolation) to full image size
    Default is None, where no elastic transformation is applied.
    :param inter_method: (optional) interpolation method when deforming the input tensor. Can be 'linear', or 'nearest'
    :param nonlin_std: (optional) maximum value of the standard deviation of the normal distribution from which we
    sample the small-size SVF.
    :param nonlin_shape_factor: (optional) ration between the shape of the input tensor and the shape of the small field
    for elastic deformation.
    :param additional_tensor: (optional) in case you want to deform another tensor with the same transformation
    :param additional_inter_method: (optional) interpolation methods for the additional tensor
    :return: tensor of the same shape as volume (a tuple, if additional tensor requested)
    """

    assert (affine_trans is not None) | apply_elastic_trans, 'affine_trans or elastic_trans should be provided'

    # reformat tensor and get its shape
    tensor = KL.Lambda(lambda x: tf.cast(x, dtype='float32'))(tensor)
    tensor._keras_shape = tuple(tensor.get_shape().as_list())
    volume_shape = tensor.get_shape().as_list()[1: -1]
    n_dims = len(volume_shape)
    trans_inputs = list()

    # add affine deformation to inputs list
    if affine_trans is not None:
        trans_inputs.append(affine_trans)

    # prepare non-linear deformation field and add it to inputs list
    if apply_elastic_trans:

        # sample small field from normal distribution of specified std dev
        small_shape = utils.get_resample_shape(volume_shape, nonlin_shape_factor, n_dims)
        tensor_shape = KL.Lambda(lambda x: tf.shape(x))(tensor)
        split_shape = KL.Lambda(lambda x: tf.split(x, [1, n_dims + 1]))(tensor_shape)
        nonlin_shape = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'), tf.convert_to_tensor(small_shape,
                                 dtype='int32')], axis=0))(split_shape[0])
        nonlin_std_prior = KL.Lambda(lambda x: tf.random.uniform((1, 1), maxval=nonlin_std))([])
        elastic_trans = KL.Lambda(lambda x: tf.random.normal(tf.cast(x[0], 'int32'),
                                                             stddev=x[1]))([nonlin_shape, nonlin_std_prior])
        elastic_trans._keras_shape = tuple(elastic_trans.get_shape().as_list())

        # reshape this field to image size and integrate it
        resize_shape = [max(int(volume_shape[i]/2), small_shape[i]) for i in range(n_dims)]
        nonlin_field = nrn_layers.Resize(size=resize_shape, interp_method='linear')(elastic_trans)
        nonlin_field = nrn_layers.VecInt()(nonlin_field)
        nonlin_field = nrn_layers.Resize(size=volume_shape, interp_method='linear')(nonlin_field)
        trans_inputs.append(nonlin_field)

    # apply deformations and return tensors
    if additional_tensor is None:
        return nrn_layers.SpatialTransformer(interp_method=inter_method)([tensor] + trans_inputs)
    else:
        additional_tensor._keras_shape = tuple(additional_tensor.get_shape().as_list())
        tens1 = nrn_layers.SpatialTransformer(interp_method=inter_method)([tensor] + trans_inputs)
        tens2 = nrn_layers.SpatialTransformer(interp_method=additional_inter_method)([additional_tensor] + trans_inputs)
        return tens1, tens2


def random_cropping(tensor, crop_shape, n_dims=3, additional_tensor=None):
    """Randomly crop an input tensor to a tensor of a given shape. This cropping is applied to all channels.
    :param tensor: input tensor to crop
    :param crop_shape: shape of the cropped tensor, excluding batch and channel dimension.
    :param n_dims: (optional) number of dimensions of the initial image (excluding batch and channel dimensions)
    :param additional_tensor: (optional) in case you want to apply the same cropping to another tensor
    :return: cropped tensor (a tuple, if additional tensor requested)
    example: if tensor has shape [2, 160, 160, 160, 3], and crop_shape=[96, 128, 96], then this function returns a
    tensor of shape [2, 96, 128, 96, 3], with randomly selected cropping indices.
    """

    # get maximum cropping indices in each dimension
    image_shape = tensor.get_shape().as_list()[1:n_dims + 1]
    cropping_max_val = [image_shape[i] - crop_shape[i] for i in range(n_dims)]

    # prepare cropping indices and tensor's new shape (don't crop batch and channel dimensions)
    crop_idx = KL.Lambda(lambda x: tf.zeros([1], dtype='int32'))([])
    for val_idx, val in enumerate(cropping_max_val):  # draw cropping indices for image dimensions
        if val > 0:
            crop_idx = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'),
                                                      tf.random.uniform([1], 0, val, 'int32')], axis=0))(crop_idx)
        else:
            crop_idx = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'),
                                                      tf.zeros([1], dtype='int32')], axis=0))(crop_idx)
    crop_idx = KL.Lambda(lambda x: tf.concat([tf.cast(x, dtype='int32'),
                                              tf.zeros([1], dtype='int32')], axis=0))(crop_idx)
    patch_shape_tens = KL.Lambda(lambda x: tf.convert_to_tensor([-1] + crop_shape + [-1], dtype='int32'))([])

    # perform cropping
    tensor = KL.Lambda(lambda x: tf.slice(x[0], begin=tf.cast(x[1], dtype='int32'),
                                              size=tf.cast(x[2], dtype='int32')))([tensor, crop_idx, patch_shape_tens])
    if additional_tensor is None:
        return tensor
    else:
        additional_tensor = KL.Lambda(lambda x: tf.slice(x[0], begin=tf.cast(x[1], dtype='int32'), size=tf.cast(x[2],
                                      dtype='int32')))([additional_tensor, crop_idx, patch_shape_tens])
        return tensor, additional_tensor


def label_map_random_flipping(labels, label_list, n_neutral_labels, aff, n_dims=3, additional_tensor=None):
    """This function flips a label map with a probability of 0.5.
    Right/left label values are also swapped if the label map is flipped in order to preserve the right/left sides.
    :param labels: input label map
    :param label_list: list of all labels contained in labels. Must be ordered as follows, first the neutral labels
    (i.e. non-sided), then left labels and right labels.
    :param n_neutral_labels: number of non-sided labels
    :param aff: affine matrix of the initial input label map, to find the right/left axis.
    :param n_dims: (optional) number of dimensions of the initial image (excluding batch and channel dimensions)
    :params additional_tensor: (optional) in case you want to apply the same flipping to another tensor. This new tensor
    is assumed to be an intensity image, thus it won't undergo any R/L value swapping.
    :return: tensor of the same shape as label map, potentially right/left flipped with correction for sided labels.
    """

    # boolean tensor to decide whether to flip
    rand_flip = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.5))([])

    # swap right and left labels if we later right-left flip the image
    n_labels = len(label_list)
    if n_neutral_labels != n_labels:
        rl_split = np.split(label_list, [n_neutral_labels, int((n_labels - n_neutral_labels) / 2 + n_neutral_labels)])
        flipped_label_list = np.concatenate((rl_split[0], rl_split[2], rl_split[1]))
        labels = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                              KL.Lambda(lambda x: tf.gather(
                                                  tf.convert_to_tensor(flipped_label_list, dtype='int32'),
                                                  tf.cast(x, dtype='int32')))(y[1]),
                                              tf.cast(y[1], dtype='int32')))([rand_flip, labels])
    # find right left axis
    ras_axes = edit_volumes.get_ras_axes(aff, n_dims)
    flip_axis = [ras_axes[0] + 1]

    # right/left flip
    labels = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                          KL.Lambda(lambda x: K.reverse(x, axes=flip_axis))(y[1]),
                                          y[1]))([rand_flip, labels])

    if additional_tensor is None:
        return labels
    else:
        additional_tensor = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                                         KL.Lambda(lambda x: K.reverse(x, axes=flip_axis))(y[1]),
                                                         y[1]))([rand_flip, additional_tensor])
        return labels, additional_tensor


def restrict_tensor(tensor, axes, boundaries):
    """Reset the edges of a tensor to zero. This is performed only along the given axes.
    The width of the zero-band is randomly drawn from a uniform distribution given in boundaries.
    :param tensor: input tensor
    :param axes: axes along which to reset edges to zero. Can be an int (single axis), or a sequence.
    :param boundaries: numpy array of shape (len(axes), 4). Each row contains the two bounds of the uniform
    distributions from which we draw the width of the zero-bands on each side.
    Those bounds must be expressed in relative side (i.e. between 0 and 1).
    :return: a tensor of the same shape as the input, with bands of zeros along the pecified axes.
    example:
    tensor=tf.constant([[[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]])  # shape = [1,10,10,1]
    axes=1
    boundaries = np.array([[0.2, 0.45, 0.85, 0.9]])

    In this case, we reset the edges along the 2nd dimension (i.e. the 1st dimension after the batch dimension),
    the 1st zero-band will expand from the 1st row to a number drawn from [0.2*tensor.shape[1], 0.45*tensor.shape[1]],
    and the 2nd zero-band will expand from a row drawn from [0.85*tensor.shape[1], 0.9*tensor.shape[1]], to the end of
    the tensor. A possible output could be:
    array([[[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]]])  # shape = [1,10,10,1]
    """

    shape = tuple(tensor.get_shape().as_list())
    axes = utils.reformat_to_list(axes, dtype='int')
    boundaries = utils.reformat_to_n_channels_array(boundaries, n_dims=4, n_channels=len(axes))

    # build mask
    mask = KL.Lambda(lambda x: tf.ones_like(x))(tensor)
    for i, axis in enumerate(axes):

        # select restricting indices
        axis_boundaries = boundaries[i, :]
        idx1 = KL.Lambda(lambda x: tf.math.round(tf.random.uniform([1], minval=axis_boundaries[0] * shape[axis],
                                                                   maxval=axis_boundaries[1] * shape[axis])))([])
        idx2 = KL.Lambda(lambda x: tf.math.round(tf.random.uniform([1], minval=axis_boundaries[2] * shape[axis],
                                                                   maxval=axis_boundaries[3] * shape[axis]) - x))(idx1)
        idx3 = KL.Lambda(lambda x: shape[axis] - x[0] - x[1])([idx1, idx2])
        split_idx = KL.Lambda(lambda x: tf.concat([x[0], x[1], x[2]], axis=0))([idx1, idx2, idx3])

        # update mask
        split_list = KL.Lambda(lambda x: tf.split(x[0], tf.cast(x[1], dtype='int32'), axis=axis))([tensor, split_idx])
        tmp_mask = KL.Lambda(lambda x: tf.concat([tf.zeros_like(x[0]), tf.ones_like(x[1]), tf.zeros_like(x[2])],
                                                 axis=axis))([split_list[0], split_list[1], split_list[2]])
        mask = KL.multiply([mask, tmp_mask])

    # mask second_channel
    tensor = KL.multiply([tensor, mask])

    return tensor, mask


def create_rigid_def_tensor(r, t, c):
    """
    This function creates a rigid deformation matrix from a 3D vector of rotations r, and 3D vector of translations t,
    and a geometric center of ratation c (essentially the center of the cuboid / image domain)
    """
    Rx_row0 = tf.constant([1, 0, 0], shape=[1, 3], dtype='float32')
    Rx_row1 = tf.stack([tf.zeros(1), tf.cos(r[tf.newaxis, 0]), -tf.sin(r[tf.newaxis, 0])], axis=1)
    Rx_row2 = tf.stack([tf.zeros(1), tf.sin(r[tf.newaxis, 0]), tf.cos(r[tf.newaxis, 0])], axis=1)
    Rx = tf.concat([Rx_row0, Rx_row1, Rx_row2], axis=0)

    Ry_row0 = tf.stack([tf.cos(r[tf.newaxis, 1]), tf.zeros(1), tf.sin(r[tf.newaxis, 1])], axis=1)
    Ry_row1 = tf.constant([0, 1, 0], shape=[1, 3], dtype='float32')
    Ry_row2 = tf.stack([-tf.sin(r[tf.newaxis, 1]), tf.zeros(1), tf.cos(r[tf.newaxis, 1])], axis=1)
    Ry = tf.concat([Ry_row0, Ry_row1, Ry_row2], axis=0)

    Rz_row0 = tf.stack([tf.cos(r[tf.newaxis, 2]), -tf.sin(r[tf.newaxis, 2]), tf.zeros(1)], axis=1)
    Rz_row1 = tf.stack([tf.sin(r[tf.newaxis, 2]), tf.cos(r[tf.newaxis, 2]), tf.zeros(1)], axis=1)
    Rz_row2 = tf.constant([0, 0, 1], shape=[1, 3], dtype='float32')
    Rz = tf.concat([Rz_row0, Rz_row1, Rz_row2], axis=0)

    R = tf.matmul(tf.matmul(Rx, Ry), Rz)

    t2 = t + c - tf.linalg.matvec(R, c)

    T = tf.concat([tf.concat([R, t2[:, tf.newaxis]], axis=1),
                   tf.constant([0, 0, 0, 1], shape=[1, 4], dtype='float32')], axis=0)

    Tinv = tf.linalg.inv(T)

    return [T[np.newaxis, :], Tinv[np.newaxis, :]]


def build_rotation_matrix(theta, n_dims):

    if n_dims == 2:
        rotation_matrix = tf.concat([tf.cos(theta), -tf.sin(theta), tf.zeros(1), tf.zeros(1),
                                     tf.sin(theta), tf.cos(theta), tf.zeros(1), tf.zeros(1),
                                     tf.zeros(1), tf.zeros(1), tf.ones(1), tf.zeros(1),
                                     tf.zeros(1), tf.zeros(1), tf.zeros(1), tf.ones(1)], axis=0)
        rotation_matrix = tf.reshape(rotation_matrix, (4, 4))
        return rotation_matrix
