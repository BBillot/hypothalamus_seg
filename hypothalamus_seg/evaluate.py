# python imports
import os
import numpy as np
from scipy.stats import wilcoxon
from scipy.ndimage.morphology import distance_transform_edt

# third-party imports
from ext.lab2im import utils
from ext.lab2im import edit_volumes


def fast_dice(x, y, labels):
    """Fast implementation of Dice scores.
    :param x: input label map
    :param y: input label map of the same size as x
    :param labels: numpy array of labels to evaluate on, sorted in increasing order.
    :return: numpy array with Dice scores in the same order as labels.
    """

    assert x.shape == y.shape, 'both inputs should have same size, had {} and {}'.format(x.shape, y.shape)

    label_edges = np.concatenate([labels[0:1] - 0.5, labels + 0.5])
    hst = np.histogram2d(x.flatten(), y.flatten(), bins=label_edges)
    c = hst[0]

    return np.diag(c) * 2 / (np.sum(c, 0) + np.sum(c, 1) + 1e-5)


def dice(x, y):
    """Implementation of dice scores ofr 0/1 numy array"""
    return 2 * np.sum(x*y) / (np.sum(x) + np.sum(y))


def surface_distances(x, y):
    """Computes the maximum boundary distance (Haussdorf distance), and the average boundary distance of two masks.
    x and y should be boolean or 0/1 numpy arrays of the same size."""

    assert x.shape == y.shape, 'both inputs should have same size, had {} and {}'.format(x.shape, y.shape)

    # detect edge
    x_dist_int = distance_transform_edt(x * 1)
    x_edge = (x_dist_int == 1) * 1
    y_dist_int = distance_transform_edt(y * 1)
    y_edge = (y_dist_int == 1) * 1

    # calculate distance from edge
    x_dist = distance_transform_edt(np.logical_not(x_edge))
    y_dist = distance_transform_edt(np.logical_not(y_edge))

    # find distances from the 2 surfaces
    x_dists_to_y = y_dist[x_edge == 1]
    y_dists_to_x = x_dist[y_edge == 1]

    # find max distance from the 2 surfaces
    if x_dists_to_y.shape[0] > 0:
        x_max_dist_to_y = np.max(x_dists_to_y)
    else:
        x_max_dist_to_y = max(x.shape)
    if y_dists_to_x.shape[0] > 0:
        y_max_dist_to_x = np.max(y_dists_to_x)
    else:
        y_max_dist_to_x = max(x.shape)

    # find average distance between 2 surfaces
    if x_dists_to_y.shape[0] > 0:
        x_mean_dist_to_y = np.mean(x_dists_to_y)
    else:
        x_mean_dist_to_y = max(x.shape)
    if y_dists_to_x.shape[0] > 0:
        y_mean_dist_to_x = np.mean(y_dists_to_x)
    else:
        y_mean_dist_to_x = max(x.shape)

    return np.maximum(x_max_dist_to_y, y_max_dist_to_x), (x_mean_dist_to_y + y_mean_dist_to_x) / 2


def compute_non_parametric_paired_test(dice_ref, dice_compare, eval_indices=None, alternative='two-sided'):
    """Compute non-parametric paired t-tests between two sets of scores.
    :param dice_ref: numpy array with scores, rows represent structures, and columns represent subjects.
    Taken as reference for one-sided tests.
    :param dice_compare: numpy array of the same format as dice_ref.
    :param eval_indices: (optional) list or 1d array indicating the row indices of structures to run the tests for
    :param alternative: (optional) The alternative hypothesis to be tested, Cab be 'two-sided', 'greater', 'less'.
    :return: 1d numpy array, with p-values for all tests on evaluated structures, as well as an additionnal test for
    average scores (last value of the array). The average score is computed only on the evaluation structures.
    """

    # take all rows if indices not specified
    if eval_indices is None:
        eval_indices = np.arange(dice_ref.shape[0])

    # loop over all evaluation label values
    pvalues = list()
    for idx in eval_indices:

        x = dice_ref[idx, :]
        y = dice_compare[idx, :]
        _, p = wilcoxon(x, y, alternative=alternative)
        pvalues.append(p)

    # average score
    x = np.mean(dice_ref[eval_indices, :], axis=0)
    y = np.mean(dice_compare[eval_indices, :], axis=0)
    _, p = wilcoxon(x, y, alternative=alternative)
    pvalues.append(p)

    return np.array(pvalues)


def reproducibility_test(gt_dir,
                         seg_dir,
                         result_dir,
                         label_list):

    """This function computes evaluation metrics (hard dice scores, average boundary distance, Hausdorff distance) 
    between two sets of labels maps in gt_dir (ground truth) and seg_dir (typically predictions).
    Labels maps are matched by sorting order.
    :param gt_dir: path of directory with gt label maps
    :param seg_dir: path of directory with label maps to compare to gt_dir. 
    The two sets of segmentations are matched by sorting order.
    :param result_dir: path of directory where result matrices will be saved.
    :param label_list: list of label values for which to compute evaluation metrics. Can be a sequence, a 1d numpy
    array, or the path to such array.
    :return: 3 matrices, each containing the results of one metric for all images.
    Rows of these matrices correspond to a different label value (in same order as in path_label_list), and each column
    represent a different subject (in hte same order as gt_dir). An additional row is added to every matrix,
    representing the score obtained by all non-zero labels.
    """

    # create result folder
    utils.mkdir(result_dir)

    # get list label maps to compare
    path_gt_labels = utils.list_images_in_folder(gt_dir)
    path_segs = utils.list_images_in_folder(seg_dir)
    if len(path_gt_labels) != len(path_segs):
        print('different number of files in data folders, had {} and {}'.format(len(path_gt_labels), len(path_segs)))

    # load labels list
    label_list = utils.get_list_labels(label_list, labels_dir=gt_dir)
    label_list_sorted = np.sort(label_list)

    # initialise result matrices
    max_dists = np.zeros((label_list.shape[0] + 1, len(path_segs)))
    mean_dists = np.zeros((label_list.shape[0] + 1, len(path_segs)))
    dice_coefs = np.zeros((label_list.shape[0] + 1, len(path_segs)))

    # loop over segmentations
    for idx, (path_seg, path_gt) in enumerate(zip(path_segs, path_gt_labels)):
        utils.print_loop_info(idx, len(path_segs), 10)

        # load gt labels and segmentation
        gt_labels = utils.load_volume(path_gt, dtype='int')
        seg = utils.load_volume(path_seg, dtype='int')
        # crop images
        gt_labels, cropping = edit_volumes.crop_volume_around_region(gt_labels, margin=10)
        seg = edit_volumes.crop_volume_with_idx(seg, cropping)
        # extract list of unique labels
        unique_gt_labels = np.unique(gt_labels)
        unique_seg_labels = np.unique(seg)
        # compute dice scores
        tmp_dice = fast_dice(gt_labels, seg, label_list_sorted)
        dice_coefs[:-1, idx] = tmp_dice[np.searchsorted(label_list_sorted, label_list)]
        # compute max/mean surface distances for all nuclei
        for index, label in enumerate(label_list):
            if (label in unique_gt_labels) & (label in unique_seg_labels):
                temp_gt = (gt_labels == label) * 1
                temp_seg = (seg == label) * 1
                max_dists[index, idx], mean_dists[index, idx] = surface_distances(temp_gt, temp_seg)
            else:
                max_dists[index, idx] = float('inf')
                mean_dists[index, idx] = float('inf')
        # compute dice, max and mean distances for whole hypothalamus
        temp_gt = (gt_labels > 0) * 1
        temp_seg = (seg > 0) * 1
        max_dists[-1, idx], mean_dists[-1, idx] = surface_distances(temp_gt, temp_seg)
        dice_coefs[-1, idx] = dice(temp_gt, temp_seg)

    # write dice and distances results
    np.save(os.path.join(result_dir, 'dice.npy'), dice_coefs)
    np.save(os.path.join(result_dir, 'max_dist.npy'), max_dists)
    np.save(os.path.join(result_dir, 'mean_dist.npy'), mean_dists)

    return dice_coefs, max_dists, mean_dists
