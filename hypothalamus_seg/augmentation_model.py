# python imports
import numpy as np
import tensorflow as tf
import keras.layers as KL
from keras.models import Model

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers
from ext.lab2im import edit_volumes
from ext.lab2im import edit_tensors as l2i_et


def build_augmentation_model(im_shape,
                             n_channels,
                             label_list,
                             image_res,
                             target_res=None,
                             output_shape=None,
                             output_div_by_n=None,
                             n_neutral_labels=1,
                             flipping=True,
                             flip_rl_only=False,
                             aff=None,
                             scaling_bounds=0.15,
                             rotation_bounds=15,
                             enable_90_rotations=False,
                             shearing_bounds=0.012,
                             translation_bounds=False,
                             nonlin_std=3.,
                             nonlin_shape_factor=.0625,
                             bias_field_std=.3,
                             bias_shape_factor=0.025,
                             same_bias_for_all_channels=False,
                             apply_intensity_augmentation=True,
                             noise_std=1.,
                             augment_channels_separately=True):

    # reformat resolutions
    im_shape = utils.reformat_to_list(im_shape)
    n_dims, _ = utils.get_dims(im_shape)
    image_res = utils.reformat_to_list(image_res, length=n_dims)
    target_res = image_res if target_res is None else utils.reformat_to_list(target_res, length=n_dims)

    # get shapes
    cropping_shape, output_shape = get_shapes(im_shape, output_shape, image_res, target_res, output_div_by_n)
    im_shape = im_shape + [n_channels]

    # create new_label_list and corresponding LUT to make sure that labels go from 0 to N-1
    new_label_list, lut = utils.rearrange_label_list(label_list)

    # define model inputs
    image_input = KL.Input(shape=im_shape, name='image_input')
    labels_input = KL.Input(shape=im_shape[:-1] + [1], name='labels_input', dtype='int32')

    # convert labels to new_label_list
    labels = l2i_et.convert_labels(labels_input, lut)

    # flipping
    if flipping:
        if flip_rl_only:
            labels, image = layers.RandomFlip(int(edit_volumes.get_ras_axes(aff, n_dims)[0]),
                                              [True, False], label_list, n_neutral_labels)([labels, image_input])
        else:
            labels, image = layers.RandomFlip(None, [True, False], label_list, n_neutral_labels)([labels, image_input])
    else:
        image = image_input

    # transform labels to soft prob. and concatenate them to the image
    labels = KL.Lambda(lambda x: tf.one_hot(tf.cast(x[..., 0], dtype='int32'), depth=len(label_list), axis=-1))(labels)
    image = KL.concatenate([image, labels], axis=len(im_shape))

    # spatial deformation
    if (scaling_bounds is not False) | (rotation_bounds is not False) | (shearing_bounds is not False) | \
       (translation_bounds is not False) | (nonlin_std > 0) | enable_90_rotations:
        image._keras_shape = tuple(image.get_shape().as_list())
        image = layers.RandomSpatialDeformation(scaling_bounds=scaling_bounds,
                                                rotation_bounds=rotation_bounds,
                                                shearing_bounds=shearing_bounds,
                                                translation_bounds=translation_bounds,
                                                enable_90_rotations=enable_90_rotations,
                                                nonlin_std=nonlin_std,
                                                nonlin_shape_factor=nonlin_shape_factor)(image)

    # cropping
    if cropping_shape != im_shape[:-1]:
        image._keras_shape = tuple(image.get_shape().as_list())
        image = layers.RandomCrop(cropping_shape)(image)

    # resampling (image blurred separately)
    if cropping_shape != output_shape:
        sigma = l2i_et.blurring_sigma_for_downsampling(image_res, target_res)
        split = KL.Lambda(lambda x: tf.split(x, [n_channels, -1], axis=len(im_shape)))(image)
        image = split[0]
        image._keras_shape = tuple(image.get_shape().as_list())
        image = layers.GaussianBlur(sigma=sigma)(image)
        image = KL.concatenate([image, split[-1]])
        image = l2i_et.resample_tensor(image, output_shape)

    # split tensor between image and labels
    image, labels = KL.Lambda(lambda x: tf.split(x, [n_channels, -1], axis=len(im_shape)), name='splitting')(image)

    # apply bias field
    if bias_field_std > 0:
        image._keras_shape = tuple(image.get_shape().as_list())
        image = layers.BiasFieldCorruption(bias_field_std, bias_shape_factor, same_bias_for_all_channels)(image)

    # intensity augmentation
    if apply_intensity_augmentation:
        image._keras_shape = tuple(image.get_shape().as_list())
        image = layers.IntensityAugmentation(noise_std, clip=False, normalise=True, norm_perc=0, gamma_std=0.5,
                                             separate_channels=augment_channels_separately)(image)

    # build model
    im_trans_model = Model(inputs=[image_input, labels_input], outputs=[image, labels])

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
