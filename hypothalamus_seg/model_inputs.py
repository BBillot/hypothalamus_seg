# python imports
import numpy as np
import numpy.random as npr

# third-party imports
from ext.lab2im import utils


def image_seg_generator(path_images,
                        path_labels,
                        batchsize=1,
                        n_channels=1,
                        apply_linear_trans=True,
                        scaling_bounds=None,
                        rotation_bounds=None,
                        shearing_bounds=None,
                        enable_90_rotations=False):

    # get image info
    _, _, n_dims, _, _, _ = utils.get_volume_info(path_images[0])

    # Generate!
    while True:

        # randomly pick as many images as batchsize
        indices = npr.randint(len(path_images), size=batchsize)

        # initialise input lists
        list_images = []
        list_label_maps = []
        list_affine_transforms = []

        for idx in indices:

            # add images and labels to inputs
            image, aff, h = utils.load_volume(path_images[idx], im_only=False, aff_ref=np.eye(4))
            label_map = utils.load_volume(path_labels[idx], dtype='int', aff_ref=np.eye(4))
            if n_channels > 1:
                list_images.append(utils.add_axis(image))
            else:
                list_images.append(utils.add_axis(image, axis=-2))
            list_label_maps.append(utils.add_axis(label_map))

            # add linear transform to inputs
            if apply_linear_trans:
                # get affine transformation: rotate, scale, shear (translation done during random cropping)
                scaling = utils.draw_value_from_distribution(scaling_bounds, size=n_dims, centre=1, default_range=.15)
                if n_dims == 2:
                    # rotation = utils.draw_value_from_distribution(rotation_bounds, default_range=15.0)
                    if enable_90_rotations:
                        centre = 90. * np.random.randint(0, 4)
                    else:
                        centre = 0
                    rotation = utils.draw_value_from_distribution(rotation_bounds, default_range=15.0, centre=centre)
                else:
                    rotation = utils.draw_value_from_distribution(rotation_bounds, size=n_dims, default_range=15.0)
                shearing = utils.draw_value_from_distribution(shearing_bounds, size=n_dims**2-n_dims, default_range=.01)
                affine_transform = utils.create_affine_transformation_matrix(n_dims, scaling, rotation, shearing)
                list_affine_transforms.append(utils.add_axis(affine_transform))

        # build list of inputs of augmentation model
        list_inputs = [list_images, list_label_maps]
        if apply_linear_trans:
            list_inputs.append(list_affine_transforms)

        # concatenate individual input types if batchsize > 1
        if batchsize > 1:
            list_inputs = [np.concatenate(item, 0) for item in list_inputs]
        else:
            list_inputs = [item[0] for item in list_inputs]

        yield list_inputs
