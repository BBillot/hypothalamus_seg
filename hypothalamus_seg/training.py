# python imports
import os
import keras
import numpy as np
import tensorflow as tf
from keras import models
import keras.callbacks as KC
from keras.optimizers import Adam
from inspect import getmembers, isclass

# project imports
from . import metrics_model as metrics
from .model_inputs import image_seg_generator
from .augmentation_model import build_augmentation_model

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers as l2i_layers
from ext.neuron import layers as nrn_layers
from ext.neuron import models as nrn_models


def training(image_dir,
             labels_dir,
             model_dir,
             path_label_list=None,
             save_label_list=None,
             n_neutral_labels=1,
             batchsize=1,
             target_res=None,
             output_shape=None,
             flipping=True,
             flip_rl_only=False,
             scaling_bounds=0.15,
             rotation_bounds=15,
             enable_90_rotations=False,
             shearing_bounds=.012,
             translation_bounds=False,
             nonlin_std=3.,
             nonlin_shape_factor=.04,
             bias_field_std=.3,
             bias_shape_factor=.025,
             same_bias_for_all_channels=False,
             augment_intensitites=True,
             noise_std=1.,
             augment_channels_separately=True,
             n_levels=5,
             nb_conv_per_level=2,
             conv_size=3,
             unet_feat_count=24,
             feat_multiplier=1,
             dropout=0,
             activation='elu',
             lr=1e-4,
             lr_decay=0,
             wl2_epochs=5,
             dice_epochs=200,
             steps_per_epoch=1000,
             checkpoint=None):
    """
    This function trains a neural network with aggressively augmented images. The model is implemented on the GPU and
    contains three sub-model: one for augmentation, one neural network (UNet), and one for computing the loss function.
    The network is pre-trained with a weighted sum of square error, in order to bring the weights in a favorable
    optimisation landscape. The training then continues with a soft dice loss function.

    :param image_dir: path of folder with all input images, or to a single image (if only one training example)
    :param labels_dir: path of folder with all input label maps, or to a single label map (if only one training example)
    labels maps and images are likend by sorting order.
    :param model_dir: path of a directory where the models will be saved during training.

    #---------------------------------------------- Generation parameters ----------------------------------------------
    # output-related parameters
    :param path_label_list: (optional) path to a numpy array containing all the label values to be segmented.
    By default, this is computed by taking all the label values in the training label maps.
    :param save_label_list: (optional) path where to write the computed list of segmentation labels.
    :param n_neutral_labels: (optional) number of non-sided labels in label_list. This is used for determining which
    label values to swap when right/left flipping the training examples. Default is 1 (to account for the background).
    :param batchsize: (optional) number of images per mini-batch. Default is 1.
    :param target_res: (optional) target resolution at which to produce the segmentation label maps. The training data
    will be resampled to this resolution before being run through the network. If None, no resampling is performed.
    Can be a number (isotropic resolution), or the path to a 1d numpy array.
    :param output_shape: (optional) desired shape of the output image, obtained by randomly cropping the generated image
    Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy array.

    # Augmentation parameters
    :param flipping: (optional) whether to introduce random flipping. Default is True.
    :param flip_rl_only: (optional) if flipping is True, whether to flip only in the right/left axis. Default is False.
    :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
    sampled from a uniform distribution of predefined bounds. scaling_bounds can either be:
    1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
    (1-scaling_bounds, 1+scaling_bounds) for each dimension.
    2) the path to a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform
    distribution of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
    3) False, in which case scaling is completely turned off.
    Default is scaling_bounds = 0.15
    :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for case 1 the
    bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
    Default is rotation_bounds = 15.
    :param enable_90_rotations: (optional) wheter to rotate the input by a random angle chosen in {0, 90, 180, 270}.
    This is done regardless of the value of rotation_bounds. If true, a different value is sampled for each dimension.
    :param shearing_bounds: (optional) same as scaling bounds. Default is shearing_bounds = 0.012.
    :param translation_bounds: (optional) same as scaling bounds. Default is translation_bounds = False, but we
    encourage using it when cropping is deactivated (i.e. when output_shape=None in BrainGenerator).
    :param nonlin_std: (optional) maximum value for the standard deviation of the normal distribution from which we
    sample the first tensor for synthesising the deformation field. Set to 0 to completely turn it off.
    :param nonlin_shape_factor: (optional) ratio between the size of the input label maps and the size of the sampled
    tensor for synthesising the elastic deformation field.
    :param bias_field_std: (optional) If strictly positive, this triggers the corruption of images with a bias field.
    It is obtained by sampling a first small tensor from a normal distribution, resizing it to full size, and rescaling
    it to positive values by taking the voxel-wise exponential. bias_field_std designates the std dev of the normal
    distribution from which we sample the first tensor. Set to 0 to deactivate biad field.
    :param bias_shape_factor: (optional) If bias_field_std is not False, this designates the ratio between the size
    of the input label maps and the size of the first sampled tensor for synthesising the bias field.
    :param same_bias_for_all_channels: (optional) If bias_field_std is not False, whether to apply the same bias field
    to all channels or not.
    :param augment_intensitites: (optional) whether to augment the intensities of the images with gamma augmentation.
    :param noise_std: (optional) if augment_intensities is true, maximum value for the standard deviation of the normal
    distribution from which we sample a Gaussian white noise. Set to False to deactivate white noise augmentation.
    Default value is 1.
    :param augment_channels_separately: (optional) whether to augment the intensities of each channel indenpendently.
    Only applied if augment_intensity is True, and the training images have several channels. Default is True.

    # ------------------------------------------ UNet architecture parameters ------------------------------------------
    :param n_levels: (optional) number of level for the Unet. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutional layers per level. Default is 2.
    :param conv_size: (optional) size of the convolution kernels. Default is 2.
    :param unet_feat_count: (optional) number of feature for the first layr of the Unet. Default is 24.
    :param feat_multiplier: (optional) multiply the number of feature by this nummber at each new level. Default is 1.
    :param dropout: (optional) probability of dropout for the Unet. Deafult is 0, where no dropout is applied.
    :param activation: (optional) activation function. Can be 'elu', 'relu'.

    # ----------------------------------------------- Training parameters ----------------------------------------------
    :param lr: (optional) learning rate for the training. Default is 1e-4
    :param lr_decay: (optional) learing rate decay. Default is 0, where no decay is applied.
    :param wl2_epochs: (optional) number of epohs for which the network (except the soft-max layer) is trained with L2
    norm loss function. Default is 5.
    :param dice_epochs: (optional) number of epochs with the soft Dice loss function. default is 100.
    :param steps_per_epoch: (optional) number of steps per epoch. Default is 1000. Since no online validation is
    possible, this is equivalent to the frequency at which the models are saved.
    :param checkpoint: (optional) path of an already saved model to load before starting the training.
    """

    # check epochs
    assert (wl2_epochs > 0) | (dice_epochs > 0), \
        'either wl2_epochs or dice_epochs must be positive, had {0} and {1}'.format(wl2_epochs, dice_epochs)

    # prepare data files
    path_images = utils.list_images_in_folder(image_dir)
    path_label_maps = utils.list_images_in_folder(labels_dir)
    assert len(path_images) == len(path_label_maps), 'not the same number of training images and label maps.'

    # read info from image and get label list
    im_shape, _, _, n_channels, _, image_res = utils.get_volume_info(path_images[0], aff_ref=np.eye(4))
    label_list, _ = utils.get_list_labels(path_label_list, labels_dir=labels_dir, save_label_list=save_label_list)
    n_labels = np.size(label_list)

    # prepare model folder
    utils.mkdir(model_dir)

    # transformation model
    augmentation_model = build_augmentation_model(im_shape=im_shape,
                                                  n_channels=n_channels,
                                                  label_list=label_list,
                                                  n_neutral_labels=n_neutral_labels,
                                                  image_res=image_res,
                                                  target_res=target_res,
                                                  output_shape=output_shape,
                                                  output_div_by_n=2 ** n_levels,
                                                  flipping=flipping,
                                                  flip_rl_only=flip_rl_only,
                                                  aff=np.eye(4),
                                                  scaling_bounds=scaling_bounds,
                                                  rotation_bounds=rotation_bounds,
                                                  enable_90_rotations=enable_90_rotations,
                                                  shearing_bounds=shearing_bounds,
                                                  translation_bounds=translation_bounds,
                                                  nonlin_std=nonlin_std,
                                                  nonlin_shape_factor=nonlin_shape_factor,
                                                  bias_field_std=bias_field_std,
                                                  bias_shape_factor=bias_shape_factor,
                                                  same_bias_for_all_channels=same_bias_for_all_channels,
                                                  apply_intensity_augmentation=augment_intensitites,
                                                  noise_std=noise_std,
                                                  augment_channels_separately=augment_channels_separately)
    unet_input_shape = augmentation_model.output[0].get_shape().as_list()[1:]

    # prepare the segmentation model
    unet_model = nrn_models.unet(nb_features=unet_feat_count,
                                 input_shape=unet_input_shape,
                                 nb_levels=n_levels,
                                 conv_size=conv_size,
                                 nb_labels=n_labels,
                                 feat_mult=feat_multiplier,
                                 nb_conv_per_level=nb_conv_per_level,
                                 conv_dropout=dropout,
                                 batch_norm=-1,
                                 activation=activation,
                                 input_model=augmentation_model)

    # input generator
    train_example_gen = image_seg_generator(path_images=path_images,
                                            path_labels=path_label_maps,
                                            batchsize=batchsize,
                                            n_channels=n_channels)
    input_generator = utils.build_training_generator(train_example_gen, batchsize)

    # pre-training with weighted L2, input is fit to the softmax rather than the probabilities
    if wl2_epochs > 0:
        wl2_model = models.Model(unet_model.inputs, [unet_model.get_layer('unet_likelihood').output])
        wl2_model = metrics.metrics_model(input_model=wl2_model, metrics='wl2')
        train_model(wl2_model, input_generator, lr, lr_decay, wl2_epochs, steps_per_epoch, model_dir, 'wl2', checkpoint)
        checkpoint = os.path.join(model_dir, 'wl2_%03d.h5' % wl2_epochs)

    # fine-tuning with dice metric
    dice_model = metrics.metrics_model(input_model=unet_model, metrics='dice')
    train_model(dice_model, input_generator, lr, lr_decay, dice_epochs, steps_per_epoch, model_dir, 'dice', checkpoint)


def train_model(model,
                generator,
                learning_rate,
                lr_decay,
                n_epochs,
                n_steps,
                model_dir,
                metric_type,
                path_checkpoint=None):

    # prepare log folder
    log_dir = os.path.join(model_dir, 'logs')
    utils.mkdir(log_dir)

    # model saving callback
    save_file_name = os.path.join(model_dir, '%s_{epoch:03d}.h5' % metric_type)
    callbacks = [KC.ModelCheckpoint(save_file_name, verbose=1)]

    # TensorBoard callback
    if metric_type == 'dice':
        callbacks.append(KC.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False))

    compile_model = True
    init_epoch = 0
    if path_checkpoint is not None:
        if metric_type in path_checkpoint:
            custom_l2i = {key: value for (key, value) in getmembers(l2i_layers, isclass) if key != 'Layer'}
            custom_nrn = {key: value for (key, value) in getmembers(nrn_layers, isclass) if key != 'Layer'}
            custom_objects = {**custom_l2i, **custom_nrn, 'tf': tf, 'keras': keras, 'loss': metrics.IdentityLoss().loss}
            model = models.load_model(path_checkpoint, custom_objects=custom_objects)
            compile_model = False
            init_epoch = int(os.path.basename(path_checkpoint).split(metric_type)[1][1:-3])
        else:
            model.load_weights(path_checkpoint, by_name=True)

    # compile
    if compile_model:
        model.compile(optimizer=Adam(lr=learning_rate, decay=lr_decay),
                      loss=metrics.IdentityLoss().loss,
                      loss_weights=[1.0])

    # fit
    model.fit_generator(generator,
                        epochs=n_epochs,
                        steps_per_epoch=n_steps,
                        callbacks=callbacks,
                        initial_epoch=init_epoch)
