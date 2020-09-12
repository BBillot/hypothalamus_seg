# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.backend as K
from keras.models import Model
from sklearn import preprocessing

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_tensors as l2i_et
from ext.lab2im import spatial_augmentation as l2i_sp
from ext.lab2im import intensity_augmentation as l2i_ia


def build_augmentation_model(im_shape,
                             n_channels,
                             label_list,
                             image_res,
                             target_res=None,
                             output_shape=None,
                             output_div_by_n=None,
                             flipping=True,
                             apply_flip_rl_only=False,
                             aff=None,
                             apply_linear_trans=True,
                             apply_nonlin_trans=True,
                             nonlin_std=3.,
                             nonlin_shape_factor=.0625,
                             apply_bias_field=True,
                             bias_field_std=.3,
                             bias_shape_factor=0.025,
                             apply_intensity_augmentation=True):

    # reformat resolutions
    im_shape = utils.reformat_to_list(im_shape)
    n_dims, _ = utils.get_dims(im_shape)
    image_res = utils.reformat_to_list(image_res, length=n_dims)
    if target_res is None:
        target_res = image_res
    else:
        target_res = utils.reformat_to_list(target_res, length=n_dims)

    # get shapes
    cropping_shape, output_shape = get_shapes(im_shape, output_shape, image_res, target_res, output_div_by_n)
    im_shape = im_shape + [n_channels]

    # create new_label_list and corresponding LUT to make sure that labels go from 0 to N-1
    n_labels = label_list.shape[0]
    new_label_list, lut = utils.rearrange_label_list(label_list)

    # define model inputs
    image_in = KL.Input(shape=im_shape, name='image_input')
    labels_in = KL.Input(shape=im_shape[:-1], name='labels_input', dtype='int32')
    list_inputs = [image_in, labels_in]
    if apply_linear_trans:
        aff_in = KL.Input(shape=(n_dims + 1, n_dims + 1), name='aff_input')
        list_inputs.append(aff_in)
    else:
        aff_in = None

    # convert labels to new_label_list
    labels = l2i_et.convert_labels(labels_in, lut)

    # flipping
    if flipping:
        # boolean tensor to decide whether to flip
        rand_flip = KL.Lambda(lambda x: K.greater(tf.random.uniform((1, 1), 0, 1), 0.5), name='bool_flip')([])
        if apply_flip_rl_only:
            # flip right and left labels if we right-left flip the image
            rl_split = np.split(new_label_list, [1, int((n_labels - 1) / 2 + 1)])
            flipped_lut = np.concatenate((rl_split[0], rl_split[2], rl_split[1]))
            labels = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                                  KL.Lambda(lambda x: tf.gather(
                                                      tf.convert_to_tensor(flipped_lut, dtype='int32'),
                                                      tf.cast(x, dtype='int32')), name='change_rl_lab')(y[1]),
                                                  tf.cast(y[1], dtype='int32')),
                               name='switch_idx_flipping')([rand_flip, labels])
            # find right left axis
            norm_vox2ras = np.absolute(preprocessing.normalize(aff[0:n_dims, 0:n_dims], axis=0))
            flip_axis_array = [np.argmax(norm_vox2ras[0, :]) + 1]  # add 1 because of batch dimension
            flip_axis = KL.Lambda(lambda x: tf.convert_to_tensor(flip_axis_array, dtype='int32'))([])
        else:
            # randomly chose an axis to flip
            flip_axis = KL.Lambda(lambda x: tf.random.uniform([1], 1, n_dims+1, dtype='int32'), name='flip_axis')([])
        # transform labels to soft prob.
        split_labels = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=n_labels, axis=-1))(labels)
        # concatenate image and labels
        concatenated = KL.concatenate([image_in, split_labels], axis=len(im_shape), name='inputs_cat')
        # flip
        image = KL.Lambda(lambda y: K.switch(tf.squeeze(y[0]),
                                             KL.Lambda(lambda x: K.reverse(x[0], axes=tf.cast(x[1], dtype='int32')),
                                                       name='flip')([y[1], y[2]]),
                                             y[1]), name='switch_im_flipping')([rand_flip, concatenated, flip_axis])
    else:
        # transform labels to soft prob.
        split_labels = KL.Lambda(lambda x: tf.one_hot(tf.cast(x, dtype='int32'), depth=n_labels, axis=-1))(labels)
        # concatenate image and labels
        image = KL.concatenate([image_in, split_labels], axis=len(im_shape), name='inputs_cat')

    # spatial deformation
    if apply_linear_trans | apply_nonlin_trans:
        image = l2i_sp.deform_tensor(image, aff_in, nonlin_std=nonlin_std, nonlin_shape_factor=nonlin_shape_factor)

    # cropping
    if cropping_shape != im_shape[:-1]:
        image, _ = l2i_sp.random_cropping(image, cropping_shape, n_dims)

    # resample image to new resolution if necessary
    if cropping_shape != output_shape:
        # separate image channels from labels channels
        split = KL.Lambda(lambda x: tf.split(x, [1]*n_channels + [-1], axis=len(im_shape)))(image)
        # blur each image channel separately
        sigma = utils.get_std_blurring_mask_for_downsampling(target_res, image_res)
        kernels_list = l2i_et.get_gaussian_1d_kernels(sigma)
        blurred_channels = list()
        for i in range(1, n_channels):
            blurred_channels.append(l2i_et.blur_tensor(split[i], kernels_list, n_dims=n_dims))
        image = KL.concatenate(blurred_channels + [split[-1]])
        # resample image at target resolution
        image = l2i_et.resample_tensor(image, output_shape)

    # split tensor between image and labels
    image, labels = KL.Lambda(lambda x: tf.split(x, [n_channels, -1], axis=len(im_shape)), name='splitting')(image)
    labels = KL.Lambda(lambda x: tf.cast(x, dtype='float32'), name='labels_out')(labels)

    # apply bias field
    if apply_bias_field:
        image = l2i_ia.bias_field_augmentation(image, bias_field_std, bias_shape_factor)

    # intensity augmentation
    if apply_intensity_augmentation:
        image = l2i_ia.min_max_normalisation(image)
        image = l2i_ia.gamma_augmentation(image, std=0.5)

    # build model
    im_trans_model = Model(inputs=list_inputs, outputs=[image, labels])

    return im_trans_model


def get_shapes(image_shape, output_shape, atlas_res, target_res, output_div_by_n):

    # reformat resolutions to lists
    atlas_res = utils.reformat_to_list(atlas_res)
    n_dims = len(atlas_res)
    target_res = utils.reformat_to_list(target_res)

    # get resampling factor
    if atlas_res != target_res:
        resample_factor = [atlas_res[i] / float(target_res[i]) for i in range(n_dims)]
    else:
        resample_factor = None

    # output shape specified, need to get cropping shape if necessary
    if output_shape is not None:
        output_shape = utils.reformat_to_list(output_shape, length=n_dims, dtype='int')

        # make sure that output shape is smaller or equal to label shape
        if resample_factor is not None:
            output_shape = [min(int(image_shape[i] * resample_factor[i]), output_shape[i]) for i in range(n_dims)]
        else:
            output_shape = [min(image_shape[i], output_shape[i]) for i in range(n_dims)]

        # make sure output shape is divisible by output_div_by_n
        if output_div_by_n is not None:
            tmp_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n, smaller_ans=True)
                         for s in output_shape]
            if output_shape != tmp_shape:
                print('output shape {0} not divisible by {1}, changed to {2}'.format(output_shape, output_div_by_n,
                                                                                     tmp_shape))
                output_shape = tmp_shape

        # get cropping and resample shape
        if resample_factor is not None:
            cropping_shape = [int(np.around(output_shape[i]/resample_factor[i], 0)) for i in range(n_dims)]
        else:
            cropping_shape = output_shape

    # no output shape specified, so no cropping unless label_shape is not divisible by output_div_by_n
    else:

        # make sure output shape is divisible by output_div_by_n
        if output_div_by_n is not None:

            # if resampling, get the potential output_shape and check if it is divisible by n
            if resample_factor is not None:
                output_shape = [int(image_shape[i] * resample_factor[i]) for i in range(n_dims)]
                output_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n, smaller_ans=True)
                                for s in output_shape]
                cropping_shape = [int(np.around(output_shape[i] / resample_factor[i], 0)) for i in range(n_dims)]
            # if no resampling, simply check if image_shape is divisible by n
            else:
                cropping_shape = [utils.find_closest_number_divisible_by_m(s, output_div_by_n, smaller_ans=True)
                                  for s in image_shape]
                output_shape = cropping_shape

        # if no need to be divisible by n, simply take cropping_shape as image_shape, and build output_shape
        else:
            cropping_shape = image_shape
            if resample_factor is not None:
                output_shape = [int(cropping_shape[i] * resample_factor[i]) for i in range(n_dims)]
            else:
                output_shape = cropping_shape

    return cropping_shape, output_shape
