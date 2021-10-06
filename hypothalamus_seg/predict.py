# python imports
import os
import csv
import numpy as np
from keras.models import Model

# project imports
from hypothalamus_seg import evaluate

# third-party imports
from ext.lab2im import utils
from ext.lab2im import layers
from ext.lab2im import edit_volumes
from ext.neuron import models as nrn_models


def predict(path_images,
            path_segmentations,
            path_model,
            segmentation_label_list,
            path_posteriors=None,
            path_resampled=None,
            path_volumes=None,
            padding=None,
            cropping=None,
            target_res=1.,
            sigma_smoothing=0,
            keep_biggest_component=False,
            conv_size=3,
            n_levels=3,
            nb_conv_per_level=2,
            unet_feat_count=24,
            feat_multiplier=2,
            activation='elu',
            gt_folder=None,
            evaluation_label_list=None,
            compute_distances=False,
            compute_score_whole_structure=True,
            recompute=True,
            verbose=True):
    """
    This function uses trained models to segment images.
    It is crucial that the inputs match the architecture parameters of the trained model.
    :param path_images: path of the images to segment. Can be the path to a directory or the path to a single image.
    :param path_segmentations: path where segmentations will be writen.
    Should be a dir, if path_images is a dir, and a file if path_images is a file.
    :param path_model: path ot the trained model.
    :param segmentation_label_list: List of labels for which to compute Dice scores. It should contain the same values
    as the segmentation label list used for training the network.
    Can be a sequence, a 1d numpy array, or the path to a numpy 1d array.
    :param path_posteriors: (optional) path where posteriors will be writen.
    Should be a dir, if path_images is a dir, and a file if path_images is a file.
    :param path_resampled: (optional) path where images resampled to 1mm isotropic will be writen.
    We emphasise that images are resampled as soon as the resolution in one of the axes is not in the range [0.9; 1.1].
    Should be a dir, if path_images is a dir, and a file if path_images is a file. Default is None, where resampled
    images are not saved.
    :param path_volumes: (optional) path of a csv file where the soft volumes of all segmented regions will be writen.
    The rows of the csv file correspond to subjects, and the columns correspond to segmentation labels.
    The soft volume of a structure corresponds to the sum of its predicted probability map.
    :param padding: (optional) pad the images to the specified shape before predicting the segmentation maps.
    Can be an int, a sequence or a 1d numpy array.
    :param cropping: (optional) crop the images to the specified shape before predicting the segmentation maps.
    If padding and cropping are specified, images are padded before being cropped.
    Can be an int, a sequence or a 1d numpy array.
    :param target_res: (optional) target resolution at which the network operates (and thus resolution of the output
    segmentations). This must match the resolution of the training data ! target_res is used to automatically resampled
    the images with resolutions outside [target_res-0.05, target_res+0.05].
    Can be a sequence, a 1d numpy array. Set to None to disable the automatic resampling. Default is 1mm.
    :param sigma_smoothing: (optional) If not None, the posteriors are smoothed with a gaussian kernel of the specified
    standard deviation.
    :param keep_biggest_component: (optional) whether to only keep the biggest component in the predicted segmentation.
    :param conv_size: (optional) size of unet's convolution masks. Default is 3.
    :param n_levels: (optional) number of levels for unet. Default is 5.
    :param nb_conv_per_level: (optional) number of convolution layers per level. Default is 2.
    :param unet_feat_count: (optional) number of features for the first layer of the unet. Default is 24.
    :param feat_multiplier: (optional) multiplicative factor for the number of feature for each new level. Default is 2.
    :param activation: (optional) activation function. Can be 'elu', 'relu'.
    :param gt_folder: (optional) path of the ground truth label maps corresponding to the input images. Should be a dir,
    if path_images is a dir, or a file if path_images is a file.
    Providing a gt_folder will trigger a Dice evaluation, where scores will be writen along with the path_segmentations.
    Specifically, the scores are contained in a numpy array, where labels are in rows, and subjects in columns.
    :param evaluation_label_list: (optional) if gt_folder is True you can evaluate the Dice scores on a subset of the
    segmentation labels, by providing another label list here. Can be a sequence, a 1d numpy array, or the path to a
    numpy 1d array. Default is the same as segmentation_label_list.
    :param compute_distances: (optional) whether to add Hausdorff and mean surface distance evaluations to the default
    Dice evaluation. Default is True.
    :param compute_score_whole_structure: (optional) whether to compute the evaluation scores for another structure
    obtained by regrouping all segmented regions. Default is True.
    :param recompute: (optional) whether to recompute segmentations that were already computed. This also applies to
    Dice scores, if gt_folder is not None. Default is True.
    :param verbose: (optional) whether to print out info about the remaining number of cases.
    """

    # prepare input/output filepaths
    path_images, path_segmentations, path_posteriors, path_resampled, path_volumes, compute = \
        prepare_output_files(path_images, path_segmentations, path_posteriors, path_resampled, path_volumes, recompute)

    # get label and classes lists
    label_list, _ = utils.get_list_labels(label_list=segmentation_label_list)
    if evaluation_label_list is None:
        evaluation_label_list = label_list

    # prepare volume file if needed
    if path_volumes is not None:
        csv_header = [['subject'] + [str(lab) for lab in label_list[1:]] + ['whole_left'] + ['whole_right']]
        with open(path_volumes, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csv_header)
        csvFile.close()

    # build network
    _, _, n_dims, n_channels, _, _ = utils.get_volume_info(path_images[0])
    model_input_shape = [None] * n_dims + [n_channels]
    net = build_model(path_model, model_input_shape, n_levels, len(label_list), conv_size,
                      nb_conv_per_level, unet_feat_count, feat_multiplier, activation, sigma_smoothing)

    # perform segmentation
    loop_info = utils.LoopInfo(len(path_images), 10, 'predicting', True)
    for idx, (path_image, path_segmentation, path_posterior, path_resample, tmp_compute) in \
            enumerate(zip(path_images, path_segmentations, path_posteriors, path_resampled, compute)):

        # compute segmentation only if needed
        if tmp_compute:
            if verbose:
                loop_info.update(idx)

            # preprocessing
            image, aff, h, im_res, _, _, shape, pad_shape, crop_idx = \
                preprocess_image(path_image, n_levels, target_res, cropping, padding, path_resample)

            # prediction
            prediction_patch = net.predict(image)

            # postprocessing
            seg, posteriors = postprocess(prediction_patch, pad_shape, shape, crop_idx, n_dims, label_list,
                                          keep_biggest_component, aff)

            # write results to disk
            if path_segmentation is not None:
                utils.save_volume(seg.astype('int'), aff, h, path_segmentation)
            if path_posterior is not None:
                if n_channels > 1:
                    posteriors = utils.add_axis(posteriors, axis=[0, -1])
                utils.save_volume(posteriors.astype('float'), aff, h, path_posterior)

        else:
            if path_volumes is not None:
                posteriors, _, _, _, _, _, im_res = utils.get_volume_info(path_posterior, True, aff_ref=np.eye(4))
            else:
                posteriors = im_res = None

        # compute volumes
        if path_volumes is not None:
            volumes = np.sum(posteriors[..., 1:], axis=tuple(range(0, len(posteriors.shape) - 1)))
            volumes = np.around(volumes * np.prod(im_res), 3)
            row = [os.path.basename(path_image).replace('.nii.gz', '')] + [str(vol) for vol in volumes]
            row += [np.sum(volumes[:int(len(volumes) / 2)]), np.sum(volumes[int(len(volumes) / 2):])]
            with open(path_volumes, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()

    # evaluate
    if gt_folder is not None:

        # find path evaluation folder
        eval_folder = os.path.dirname(path_segmentations[0])

        # set path of result arrays for surface distance if necessary
        if compute_distances:
            path_hausdorff = os.path.join(eval_folder, 'hausdorff.npy')
            path_mean_distance = os.path.join(eval_folder, 'mean_distance.npy')
        else:
            path_hausdorff = path_mean_distance = None

        # compute evaluation metrics
        evaluate.evaluation(gt_folder,
                            eval_folder,
                            evaluation_label_list,
                            compute_score_whole_structure=compute_score_whole_structure,
                            path_dice=os.path.join(eval_folder, 'dice.npy'),
                            path_hausdorff=path_hausdorff,
                            path_mean_distance=path_mean_distance,
                            recompute=recompute,
                            verbose=verbose)


def prepare_output_files(path_images, out_seg, out_posteriors, out_resampled, out_volumes, recompute):
    '''
    Prepare output files.
    '''

    # check inputs
    assert out_seg is not None, 'please specify an output file/folder (--o)'

    # convert path to absolute paths
    path_images = os.path.abspath(path_images)
    basename = os.path.basename(path_images)
    out_seg = os.path.abspath(out_seg) if (out_seg is not None) else out_seg
    out_posteriors = os.path.abspath(out_posteriors) if (out_posteriors is not None) else out_posteriors
    out_resampled = os.path.abspath(out_resampled) if (out_resampled is not None) else out_resampled
    out_volumes = os.path.abspath(out_volumes) if (out_volumes is not None) else out_volumes

    # path_images is a folder
    if ('.nii.gz' not in basename) & ('.nii' not in basename) & ('.mgz' not in basename) & ('.npz' not in basename):
        if os.path.isfile(path_images):
            raise Exception('Extension not supported for %s, only use: nii.gz, .nii, .mgz, or .npz' % path_images)
        path_images = utils.list_images_in_folder(path_images)
        utils.mkdir(out_seg)
        out_seg = [os.path.join(out_seg, os.path.basename(image)).replace('.nii', '_hypo_seg.nii') for image in
                   path_images]
        out_seg = [seg_path.replace('.mgz', '_hypo_seg.mgz') for seg_path in out_seg]
        out_seg = [seg_path.replace('.npz', '_hypo_seg.npz') for seg_path in out_seg]
        recompute_seg = [not os.path.isfile(path_seg) for path_seg in out_seg]
        if out_posteriors is not None:
            utils.mkdir(out_posteriors)
            out_posteriors = [os.path.join(out_posteriors, os.path.basename(image)).replace('.nii',
                              '_posteriors.nii') for image in path_images]
            out_posteriors = [posteriors_path.replace('.mgz', '_posteriors.mgz') for posteriors_path in out_posteriors]
            out_posteriors = [posteriors_path.replace('.npz', '_posteriors.npz') for posteriors_path in out_posteriors]
            recompute_post = [not os.path.isfile(path_post) for path_post in out_posteriors]
        else:
            out_posteriors = [out_posteriors] * len(path_images)
            recompute_post = [out_volumes is not None] * len(path_images)
        if out_resampled is not None:
            utils.mkdir(out_resampled)
            out_resampled = [os.path.join(out_resampled, os.path.basename(image)).replace('.nii',
                             '_resampled.nii') for image in path_images]
            out_resampled = [resampled_path.replace('.mgz', '_resampled.mgz') for resampled_path in out_resampled]
            out_resampled = [resampled_path.replace('.npz', '_resampled.npz') for resampled_path in out_resampled]
            recompute_resampled = [not os.path.isfile(path_post) for path_post in out_resampled]
        else:
            out_resampled = [out_resampled] * len(path_images)
            recompute_resampled = [out_volumes is not None] * len(path_images)

    # path_images is an image
    else:
        assert os.path.isfile(path_images), "file does not exist: %s \n" \
                                            "please make sure the path and the extension are correct" % path_images
        path_images = [path_images]
        if ('.nii.gz' not in out_seg) & ('.nii' not in out_seg) & ('.mgz' not in out_seg) & ('.npz' not in out_seg):
            utils.mkdir(out_seg)
            filename = os.path.basename(path_images[0]).replace('.nii', '_hypo_seg.nii')
            filename = filename.replace('.mgz', '_hypo_seg.mgz')
            filename = filename.replace('.npz', '_hypo_seg.npz')
            out_seg = os.path.join(out_seg, filename)
        else:
            utils.mkdir(os.path.dirname(out_seg))
        out_seg = [out_seg]
        recompute_seg = [not os.path.isfile(out_seg[0])]
        if out_posteriors is not None:
            if ('.nii.gz' not in out_posteriors) & ('.nii' not in out_posteriors) &\
                    ('.mgz' not in out_posteriors) & ('.npz' not in out_posteriors):
                utils.mkdir(out_posteriors)
                filename = os.path.basename(path_images[0]).replace('.nii', '_posteriors.nii')
                filename = filename.replace('.mgz', '_posteriors.mgz')
                filename = filename.replace('.npz', '_posteriors.npz')
                out_posteriors = os.path.join(out_posteriors, filename)
            else:
                utils.mkdir(os.path.dirname(out_posteriors))
            recompute_post = [not os.path.isfile(out_posteriors[0])]
        else:
            recompute_post = [out_volumes is not None]
        out_posteriors = [out_posteriors]
        if out_resampled is not None:
            if ('.nii.gz' not in out_resampled) & ('.nii' not in out_resampled) &\
                    ('.mgz' not in out_resampled) & ('.npz' not in out_resampled):
                utils.mkdir(out_resampled)
                filename = os.path.basename(path_images[0]).replace('.nii', '_resampled.nii')
                filename = filename.replace('.mgz', '_resampled.mgz')
                filename = filename.replace('.npz', '_resampled.npz')
                out_resampled = os.path.join(out_resampled, filename)
            else:
                utils.mkdir(os.path.dirname(out_resampled))
            recompute_resampled = [not os.path.isfile(out_resampled[0])]
        else:
            recompute_resampled = [out_volumes is not None]
        out_resampled = [out_resampled]

    recompute_list = [recompute | re_seg | re_post | re_res
                      for (re_seg, re_post, re_res) in zip(recompute_seg, recompute_post, recompute_resampled)]

    if out_volumes is not None:
        if out_volumes[-4:] != '.csv':
            print('Path for volume outputs provided without csv extension. Adding csv extension.')
            out_volumes += '.csv'
            utils.mkdir(os.path.dirname(out_volumes))

    return path_images, out_seg, out_posteriors, out_resampled, out_volumes, recompute_list


def preprocess_image(im_path, n_levels, target_res, crop=None, padding=None, path_resample=None):

    # read image and corresponding info
    im, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(im_path, True)

    # resample image if necessary
    if target_res is not None:
        target_res = np.squeeze(utils.reformat_to_n_channels_array(target_res, n_dims))
        if np.any((im_res > target_res + 0.05) | (im_res < target_res - 0.05)):
            im_res = target_res
            im, aff = edit_volumes.resample_volume(im, aff, im_res)
            shape = list(im.shape)
            if path_resample is not None:
                utils.save_volume(im, aff, header, path_resample)

    # align image
    im = edit_volumes.align_volume_to_ref(im, aff, aff_ref=np.eye(4), n_dims=n_dims)

    # pad image if specified
    if padding:
        im = edit_volumes.pad_volume(im, padding_shape=padding)
        pad_shape = im.shape[:n_dims]
    else:
        pad_shape = shape

    # check that patch_shape or im_shape are divisible by 2**n_levels
    if crop is not None:
        crop = utils.reformat_to_list(crop, length=n_dims, dtype='int')
        if not all([pad_shape[i] >= crop[i] for i in range(len(pad_shape))]):
            crop = [min(pad_shape[i], crop[i]) for i in range(n_dims)]
        if not all([size % (2**n_levels) == 0 for size in crop]):
            crop = [utils.find_closest_number_divisible_by_m(size, 2 ** n_levels) for size in crop]
    else:
        if not all([size % (2**n_levels) == 0 for size in pad_shape]):
            crop = [utils.find_closest_number_divisible_by_m(size, 2 ** n_levels) for size in pad_shape]

    # crop image if necessary
    if crop is not None:
        im, crop_idx = edit_volumes.crop_volume(im, cropping_shape=crop, return_crop_idx=True)
    else:
        crop_idx = None

    # normalise image
    if n_channels == 1:
        im = edit_volumes.rescale_volume(im, new_min=0., new_max=1., min_percentile=0.5, max_percentile=99.5)
    else:
        for i in range(im.shape[-1]):
            im[..., i] = edit_volumes.rescale_volume(im[..., i], new_min=0., new_max=1.,
                                                     min_percentile=0.5, max_percentile=99.5)

    # add batch and channel axes
    im = utils.add_axis(im) if n_channels > 1 else utils.add_axis(im, axis=[0, -1])

    return im, aff, header, im_res, n_channels, n_dims, shape, pad_shape, crop_idx


def build_model(model_file, input_shape, n_levels, n_lab, conv_size, nb_conv_per_level, unet_feat_count,
                feat_multiplier, activation, sigma_smoothing):

    assert os.path.isfile(model_file), "The provided model path does not exist."

    # build UNet
    net = nrn_models.unet(nb_features=unet_feat_count,
                          input_shape=input_shape,
                          nb_levels=n_levels,
                          conv_size=conv_size,
                          nb_labels=n_lab,
                          feat_mult=feat_multiplier,
                          activation=activation,
                          nb_conv_per_level=nb_conv_per_level,
                          batch_norm=-1)
    net.load_weights(model_file, by_name=True)

    # smooth posteriors if specified
    if sigma_smoothing > 0:
        last_tensor = net.output
        last_tensor._keras_shape = tuple(last_tensor.get_shape().as_list())
        last_tensor = layers.GaussianBlur(sigma=sigma_smoothing)(last_tensor)
        net = Model(inputs=net.inputs, outputs=last_tensor)

    return net


def postprocess(post_patch, pad_shape, im_shape, crop, n_dims, labels, keep_biggest_component, aff):

    # get posteriors and segmentation
    post_patch = np.squeeze(post_patch)
    seg_patch = post_patch.argmax(-1)

    # keep biggest connected component (use it with smoothing!)
    if keep_biggest_component:
        left_mask = edit_volumes.get_largest_connected_component((seg_patch > 0) & (seg_patch < 6))
        right_mask = edit_volumes.get_largest_connected_component(seg_patch > 5)
        seg_patch *= (left_mask | right_mask)

    # paste patches back to matrix of original image size
    if crop is not None:
        seg = np.zeros(shape=pad_shape, dtype='int32')
        posteriors = np.zeros(shape=[*pad_shape, labels.shape[0]])
        posteriors[..., 0] = np.ones(pad_shape)  # place background around patch
        if n_dims == 2:
            seg[crop[0]:crop[2], crop[1]:crop[3]] = seg_patch
            posteriors[crop[0]:crop[2], crop[1]:crop[3], :] = post_patch
        elif n_dims == 3:
            seg[crop[0]:crop[3], crop[1]:crop[4], crop[2]:crop[5]] = seg_patch
            posteriors[crop[0]:crop[3], crop[1]:crop[4], crop[2]:crop[5], :] = post_patch
    else:
        seg = seg_patch
        posteriors = post_patch
    seg = labels[seg.astype('int')].astype('int')

    if im_shape != pad_shape:
        bounds = [int((p-i)/2) for (p, i) in zip(pad_shape, im_shape)]
        bounds += [p + i for (p, i) in zip(bounds, im_shape)]
        seg = edit_volumes.crop_volume_with_idx(seg, bounds)
        posteriors = edit_volumes.crop_volume_with_idx(posteriors, bounds, n_dims=n_dims)

    # align prediction back to first orientation
    if n_dims > 2:
        seg = edit_volumes.align_volume_to_ref(seg, aff=np.eye(4), aff_ref=aff, n_dims=n_dims, return_aff=False)
        posteriors = edit_volumes.align_volume_to_ref(posteriors, aff=np.eye(4), aff_ref=aff, n_dims=n_dims)

    return seg, posteriors
