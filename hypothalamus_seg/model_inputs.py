# python imports
import numpy as np
import numpy.random as npr

# third-party imports
from ext.lab2im import utils


def image_seg_generator(path_images,
                        path_label_maps,
                        batchsize=1,
                        n_channels=1):
    """
    This function builds a generator that will be used to give the necessary inputs to the augmentation model (i.e. the
    input image and label map).
    :param path_images: list of the paths of the input images.
    :param path_label_maps: list of the paths of the input label maps (in the same order as path_images).
    :param batchsize: (optional) numbers of examples per mini-batch. Default is 1.
    :param n_channels: (optional) number of channels of the input images. Default is 1.
    """

    # get image info
    _, _, n_dims, _, _, _ = utils.get_volume_info(path_images[0])

    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = npr.randint(len(path_images), size=batchsize)

        # initialise input lists
        list_images = []
        list_label_maps = []

        for idx in indices:

            # add images and labels to inputs
            image = utils.load_volume(path_images[idx], aff_ref=np.eye(4))
            if n_channels > 1:
                list_images.append(utils.add_axis(image))
            else:
                list_images.append(utils.add_axis(image, axis=[0, -1]))

            # add labels
            label_map = utils.load_volume(path_label_maps[idx], dtype='int', aff_ref=np.eye(4))
            list_label_maps.append(utils.add_axis(label_map, axis=[0, -1]))

        # build list of inputs of augmentation model
        list_inputs = [list_images, list_label_maps]
        if batchsize > 1:  # concatenate each input type if batchsize > 1
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs
