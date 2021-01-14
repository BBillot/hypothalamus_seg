# python imports
import numpy as np
import numpy.random as npr

# third-party imports
from ext.lab2im import utils


def image_seg_generator(path_images,
                        path_labels,
                        batchsize=1,
                        n_channels=1):

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
            label_map = utils.load_volume(path_labels[idx], dtype='int', aff_ref=np.eye(4))
            list_label_maps.append(utils.add_axis(label_map, axis=[0, -1]))

        # build list of inputs of augmentation model
        list_inputs = [list_images, list_label_maps]
        if batchsize > 1:  # concatenate individual input types if batchsize > 1
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs
