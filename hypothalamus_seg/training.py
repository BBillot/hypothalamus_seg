# python imports
import os
import numpy as np
import keras.callbacks as KC
from keras.models import Model
from keras.optimizers import Adam

# project imports
from . import metrics_model
from .model_inputs import image_seg_generator
from .augmentation_model import build_augmentation_model

# third-party imports
from ext.lab2im import utils
from ext.neuron import models as nrn_models


def training(image_dir,
             labels_dir,
             model_dir,
             path_label_list=None,
             save_label_list=None,
             batchsize=1,
             target_res=None,
             output_shape=None,
             flipping=True,
             flip_rl_only=False,
             apply_linear_trans=True,
             scaling_bounds=0.1,
             rotation_bounds=15,
             shearing_bounds=.012,
             apply_nonlin_trans=True,
             nonlin_std=3.,
             nonlin_shape_factor=.04,
             apply_bias_field=True,
             bias_field_std=.3,
             bias_shape_factor=.025,
             augment_intensitites=True,
             n_levels=5,
             nb_conv_per_level=2,
             conv_size=3,
             unet_feat_count=24,
             feat_multiplier=1,
             dropout=0,
             no_batch_norm=False,
             lr=1e-4,
             lr_decay=0,
             wl2_epochs=5,
             dice_epochs=200,
             steps_per_epoch=1000,
             background_weight=1e-4,
             include_background=True,
             load_model_file=None,
             initial_epoch_wl2=0,
             initial_epoch_dice=0):
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
    :param batchsize: (optional) number of images per mini-batch. Default is 1.
    :param target_res: (optional) target resolution at which to produce the segmentation label maps. The training data
    will be resampled to this resolution before being run through the network. If None, no resampling is performed.
    Can be a number (isotropic resolution), or the path to a 1d numpy array.
    :param output_shape: (optional) desired shape of the output image, obtained by randomly cropping the generated image
    Can be an integer (same size in all dimensions), a sequence, a 1d numpy array, or the path to a 1d numpy array.

    # Augmentation parameters
    :param flipping: (optional) whether to introduce random flipping. Default is True.
    :param flip_rl_only: (optional) if flipping is True, whether to flip only in the right/left axis. Default is False.
    :param apply_linear_trans: (optional) whether to apply a random linear transformation to the training data.
    This includes random scaling, rotation, and shearing. Default is True.
    :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
    sampled from a uniform distribution of predefined bounds. scaling_bounds can either be:
    1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
    (1-scaling_bounds, 1+scaling_bounds) for each dimension.
    2) the path to a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform
    distribution of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
    If None (default), scaling_range = 0.15
    :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for case 1 the
    bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
    If None (default), rotation_bounds = 15.
    :param shearing_bounds: (optional) same as scaling bounds. If None (default), shearing_bounds = 0.01.
    :param apply_nonlin_trans: (optional) whether to apply a random elastic deformation to the training data.
    This is done by sampling a small non linear field of size batch*(dim_1*...*dim_n)*n_dims from a Gaussian
    distribution. This field is then resampled to image size, and finally integrated to obtain a diffeomorphic elastic
    deformation. Default is True.
    :param nonlin_std: (optional) if apply_nonlin_trans is True, maximum value for the standard deviation of the normal
    distribution from which we sample the first tensor for synthesising the deformation field.
    :param nonlin_shape_factor: (optional) ratio between the size of the input label maps and the size of the sampled
    tensor for synthesising the elastic deformation field.
    :param apply_bias_field: (optional) whether to apply a random bias field to the training image. This is done by
    sampling a small field of size batch*(dim_1*...*dim_n), which is then resampled to image size, and rescaled to
    postive values by taking the voxel-wise exponential. This field is finally multiplied to the image.
    :param bias_field_std: (optional) if apply_bias_field is True, maximum value for the standard deviation of the
    normal distribution from which we sample the first tensor for synthesising the bias field.
    :param bias_shape_factor: (optional) Ratio between the size of the input label maps and the size of the sampled
    tensor for synthesising the bias field.
    :param augment_intensitites: (optional) whether to augment the intensities of the images with gamma augmentation.

    # ------------------------------------------ UNet architecture parameters ------------------------------------------
    :param n_levels: (optional) number of level for the Unet. Default is 5.
    :param nb_conv_per_level: (optional) number of convolutional layers per level. Default is 2.
    :param conv_size: (optional) size of the convolution kernels. Default is 2.
    :param unet_feat_count: (optional) number of feature for the first layr of the Unet. Default is 24.
    :param feat_multiplier: (optional) multiply the number of feature by this nummber at each new level. Default is 1.
    :param dropout: (optional) probability of drpout for the Unet. Deafult is 0, where no dropout is applied.
    :param no_batch_norm: (optional) wheter to remove batch normalisation. Default is False, where BN is applied.

    # ----------------------------------------------- Training parameters ----------------------------------------------
    :param lr: (optional) learning rate for the training. Default is 1e-4
    :param lr_decay: (optional) learing rate decay. Default is 0, where no decay is applied.
    :param wl2_epochs: (optional) number of epohs for which the network (except the soft-max layer) is trained with L2
    norm loss function. Default is 5.
    :param dice_epochs: (optional) number of epochs with the soft Dice loss function. default is 100.
    :param steps_per_epoch: (optional) number of steps per epoch. Default is 1000. Since no online validation is
    possible, this is equivalent to the frequency at which the models are saved.
    :param background_weight: (optional) weight of the background when computing loss. Default is 1e-4.
    :param include_background: (optional) whether to include Dice of background when evaluating the loss function.
    :param load_model_file: (optional) path of an already saved model to load before starting the training.
    :param initial_epoch_wl2: (optional) initial epoch for wl2 training. Useful for resuming training.
    :param initial_epoch_dice: (optional) initial epoch for dice training. Useful for resuming training.
    """

    # check epochs
    assert (wl2_epochs > 0) | (dice_epochs > 0), \
        'either wl2_epochs or dice_epochs must be positive, had {0} and {1}'.format(wl2_epochs, dice_epochs)

    # prepare data files
    if ('.nii.gz' in image_dir) | ('.nii' in image_dir) | ('.mgz' in image_dir) | ('.npz' in image_dir):
        assert os.path.isfile(image_dir), 'no such file: %s' % image_dir
        path_images = [image_dir]
    else:
        path_images = utils.list_images_in_folder(image_dir)
    if ('.nii.gz' in labels_dir) | ('.nii' in labels_dir) | ('.mgz' in labels_dir) | ('.npz' in labels_dir):
        assert os.path.isfile(labels_dir), 'no such file: %s' % labels_dir
        path_label_maps = [labels_dir]
    else:
        path_label_maps = utils.list_images_in_folder(labels_dir)
    assert len(path_images) == len(path_label_maps), 'not the same number of training images and label maps.'

    # read info from image and get label list
    im_shape, aff, _, n_channels, _, image_res = utils.get_volume_info(path_images[0])
    label_list = utils.get_list_labels(path_label_list, labels_dir=labels_dir, save_label_list=save_label_list)
    n_labels = np.size(label_list)

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # prepare log folder
    log_dir = os.path.join(model_dir, 'logs')
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    # transformation model
    augmentation_model = build_augmentation_model(im_shape=im_shape,
                                                  n_channels=n_channels,
                                                  label_list=label_list,
                                                  image_res=image_res,
                                                  target_res=target_res,
                                                  output_shape=output_shape,
                                                  output_div_by_n=2**n_levels,
                                                  flipping=flipping,
                                                  apply_flip_rl_only=flip_rl_only,
                                                  aff=aff,
                                                  apply_linear_trans=apply_linear_trans,
                                                  apply_nonlin_trans=apply_nonlin_trans,
                                                  nonlin_std=nonlin_std,
                                                  nonlin_shape_factor=nonlin_shape_factor,
                                                  apply_bias_field=apply_bias_field,
                                                  bias_field_std=bias_field_std,
                                                  bias_shape_factor=bias_shape_factor,
                                                  apply_intensity_augmentation=augment_intensitites)
    unet_input_shape = augmentation_model.output[0].get_shape().as_list()[1:]

    # prepare the segmentation model
    if no_batch_norm:
        batch_norm_dim = None
    else:
        batch_norm_dim = -1
    unet_model = nrn_models.unet(nb_features=unet_feat_count,
                                 input_shape=unet_input_shape,
                                 nb_levels=n_levels,
                                 conv_size=conv_size,
                                 nb_labels=n_labels,
                                 feat_mult=feat_multiplier,
                                 nb_conv_per_level=nb_conv_per_level,
                                 conv_dropout=dropout,
                                 batch_norm=batch_norm_dim,
                                 input_model=augmentation_model)

    # input generator
    train_example_gen = image_seg_generator(path_images=path_images,
                                            path_labels=path_label_maps,
                                            batchsize=batchsize,
                                            n_channels=n_channels,
                                            apply_linear_trans=apply_linear_trans,
                                            scaling_bounds=scaling_bounds,
                                            rotation_bounds=rotation_bounds,
                                            shearing_bounds=shearing_bounds)
    training_generator = utils.build_training_generator(train_example_gen, batchsize)

    # pre-training with weighted L2, input is fit to the softmax rather than the probabilities
    if wl2_epochs > 0:
        wl2_model = Model(unet_model.inputs, [unet_model.get_layer('unet_likelihood').output])
        wl2_model = metrics_model.metrics_model(input_shape=unet_input_shape[:-1] + [n_labels],
                                                input_model=wl2_model,
                                                metrics='weighted_l2',
                                                weight_background=background_weight,
                                                name='metrics_model')
        if load_model_file is not None:
            print('loading', load_model_file)
            wl2_model.load_weights(load_model_file)
        train_model(wl2_model, training_generator, lr, lr_decay, wl2_epochs, steps_per_epoch, model_dir, log_dir,
                    'wl2', initial_epoch_wl2)

    # fine-tuning with dice metric
    if dice_epochs > 0:
        dice_model = metrics_model.metrics_model(input_shape=unet_input_shape[:-1] + [n_labels],
                                                 input_model=unet_model,
                                                 include_background=include_background,
                                                 name='metrics_model')
        if wl2_epochs > 0:
            last_wl2_model_name = os.path.join(model_dir, 'wl2_%03d.h5' % wl2_epochs)
            dice_model.load_weights(last_wl2_model_name, by_name=True)
        elif load_model_file is not None:
            print('loading', load_model_file)
            dice_model.load_weights(load_model_file)
        train_model(dice_model, training_generator, lr, lr_decay, dice_epochs, steps_per_epoch, model_dir, log_dir,
                    'dice', initial_epoch_dice)


def train_model(model,
                generator,
                learn_rate,
                lr_decay,
                n_epochs,
                n_steps,
                model_dir,
                log_dir,
                metric_type,
                initial_epoch=0):

    # model saving callback
    save_file_name = os.path.join(model_dir, '%s_{epoch:03d}.h5' % metric_type)
    callbacks = [KC.ModelCheckpoint(save_file_name, save_weights_only=True, verbose=1)]

    # TensorBoard callback
    if metric_type == 'dice':
        callbacks.append(KC.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False))

    # decrease learning rate by 1 magnitude after 350 epochs
    def scheduler(epoch, lr):
        if epoch == 350 - 1:
            return lr / 10
        else:
            return lr
    callbacks.append(KC.LearningRateScheduler(scheduler))

    # compile
    model.compile(optimizer=Adam(lr=learn_rate, decay=lr_decay),
                  loss=metrics_model.IdentityLoss().loss,
                  loss_weights=[1.0])

    # fit
    model.fit_generator(generator,
                        epochs=n_epochs,
                        steps_per_epoch=n_steps,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch)
