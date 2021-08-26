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
                             bias_field_std=.5,
                             bias_shape_factor=.025,
                             same_bias_for_all_channels=False,
                             apply_intensity_augmentation=True,
                             noise_std=1.,
                             augment_channels_separately=True):
    """
    This function builds a keras/tensorflow model to perform spatial and intensity augmentation of the input image and
    labels map. The model returns:
        -the augmented image normalised between 0 and 1.
        -the corresponding label map, under the form of soft probability maps for each label in label_list.
    # IMPORTANT !!!
    # Each time we provide a parameter with separate values for each axis (e.g. with a numpy array or a sequence),
    # these values refer to the RAS axes.
    :param im_shape: shape of the input images. Can be a sequence or a 1d numpy array.
    :param n_channels: number of channels of the input images.
    :param label_list: (optional) list of the label values to be segmented.
    If not None, can be a sequence or a 1d numpy array. It should be organised as follows: background label first, then
    non-sided labels (e.g. CSF, brainstem, etc.), then all the structures of the same hemisphere (can be left or right),
    and finally all the corresponding contralateral structures (in the same order).
    :param target_res: resolution at which to resample the images and corresponding label maps.
    Default is None, where no resampling is performed.
    :param output_shape: (optional) desired shape of the outputs (image and label map), obtained by random cropping.
    Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy array.
    Default is None, where no cropping is performed.
    :param output_div_by_n: (optional) forces the output shape to be divisible by this value. It overwrites output_shape
    if necessary. Can be an integer (same size in all dimensions), a sequence, or a 1d numpy array.
    :param n_neutral_labels: number of non-sided labels (including background).
    Can be a number (isotropic resolution), a sequence, or a 1d numpy array.
    :param flipping: (optional) whether to introduce right/left random flipping
    :param flip_rl_only: (optional) if flipping is True, whether to flip only in the right/left axis. Default is False.
    :param aff: (optional) example of an (n_dims+1)x(n_dims+1) affine matrix of one of the input label map.
    Used to find brain's right/left axis. Should be given if flipping is True.
    :param scaling_bounds: (optional) range of the random scaling to apply at each mini-batch. The scaling factor for
    each dimension is sampled from a uniform distribution of predefined bounds. Can either be:
    1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
    [1-scaling_bounds, 1+scaling_bounds] for each dimension.
    2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
    (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
    3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
     of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
    4) False, in which case scaling is completely turned off.
    Default is scaling_bounds = 0.15 (case 1)
    :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
    and 2, the bounds are centred on 0 rather than 1, i.e. [0+rotation_bounds[i], 0-rotation_bounds[i]].
    Default is rotation_bounds = 15.
    :param enable_90_rotations: (optional) wheter to rotate the input by a random angle chosen in {0, 90, 180, 270}.
    This is done regardless of the value of rotation_bounds. If true, a different value is sampled for each dimension.
    :param shearing_bounds: (optional) same as scaling bounds. Default is shearing_bounds = 0.012.
    :param translation_bounds: (optional) same as scaling bounds. Default is translation_bounds = False, but we
    encourage using it when cropping is deactivated (i.e. when output_shape=None in BrainGenerator).
    :param nonlin_std: (optional) Maximum value for the standard deviation of the normal distribution from which we
    sample the first tensor for synthesising the deformation field. Set to 0 if you wish to completely turn the elastic
    deformation off.
    :param nonlin_shape_factor: (optional) if nonlin_std is strictly positive, factor between the shapes of the input
    label maps and the shape of the input non-linear tensor.
    :param bias_field_std: (optional) If strictly positive, this triggers the corruption of synthesised images with a
    bias field. It is obtained by sampling a first small tensor from a normal distribution, resizing it to full size,
    and rescaling it to positive values by taking the voxel-wise exponential. bias_field_std designates the std dev of
    the normal distribution from which we sample the first tensor. Set to 0 to deactivate biad field corruption.
    :param bias_shape_factor: (optional) If bias_field_std is strictly positive, this designates the ratio between the
    size of the input label maps and the size of the first sampled tensor for synthesising the bias field.
    :param same_bias_for_all_channels: (optional) If bias_field_std is not False, whether to apply the same bias field
    to all channels or not.
    :param augment_intensitites: (optional) whether to augment the intensities of the images with gamma augmentation.
    :param noise_std: (optional) if augment_intensities is True, maximum value for the standard deviation of the normal
    distribution from which we sample a Gaussian white noise. Set to False to deactivate white noise augmentation.
    Default value is 1.
    :param augment_channels_separately: (optional) whether to augment the intensities of each channel indenpendently.
    Only applied if augment_intensity is True, and the training images have several channels. Default is True.
    """

    # reformat resolutions
    im_shape = utils.reformat_to_list(im_shape)
    n_dims, _ = utils.get_dims(im_shape)
    image_res = utils.reformat_to_list(image_res, n_dims)
    target_res = image_res if (target_res is None) else utils.reformat_to_list(target_res, n_dims)

    # get shapes
    crop_shape, output_shape = get_shapes(im_shape, output_shape, image_res, target_res, output_div_by_n)
    im_shape = im_shape + [n_channels]

    # define model inputs
    image_input = KL.Input(shape=im_shape, name='image_input')
    labels_input = KL.Input(shape=im_shape[:-1] + [1], name='labels_input', dtype='int32')

    # flipping
    if flipping:
        rl_axis = edit_volumes.get_ras_axes(aff, n_dims)[0] if flip_rl_only else None
        labels, image = layers.RandomFlip(rl_axis, [True, False],
                                          label_list, n_neutral_labels)([labels_input, image_input])
    else:
        labels = labels_input
        image = image_input

    # transform labels to soft prob. and concatenate them to the image
    labels = KL.Lambda(lambda x: tf.one_hot(tf.cast(x[..., 0], dtype='int32'), depth=len(label_list), axis=-1))(labels)
    image = KL.concatenate([image, labels], axis=len(im_shape))

    # spatial deformation
    image._keras_shape = tuple(image.get_shape().as_list())
    image = layers.RandomSpatialDeformation(scaling_bounds=scaling_bounds,
                                            rotation_bounds=rotation_bounds,
                                            shearing_bounds=shearing_bounds,
                                            translation_bounds=translation_bounds,
                                            enable_90_rotations=enable_90_rotations,
                                            nonlin_std=nonlin_std,
                                            nonlin_shape_factor=nonlin_shape_factor)(image)

    # cropping
    if crop_shape != im_shape[:-1]:
        image._keras_shape = tuple(image.get_shape().as_list())
        image = layers.RandomCrop(crop_shape)(image)

    # resampling (the image is blurred separately before resampling)
    if crop_shape != output_shape:
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
        image = layers.IntensityAugmentation(noise_std, gamma_std=.5,
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

    # output shape specified, need to get cropping shape, and resample shape if necessary
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
