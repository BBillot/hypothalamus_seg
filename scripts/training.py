from argparse import ArgumentParser
from hypothalamus_seg.training import training
from ext.lab2im.utils import infer

parser = ArgumentParser()

# Positional arguments
parser.add_argument("image_dir", type=str, help="directory containing training images")
parser.add_argument("labels_dir", type=str, help="directory containing corresponding training label maps")
parser.add_argument("model_dir", type=str, help="directory where models will be saved")

# Output-related parameters
parser.add_argument("--label_list", type=str, default=None, dest="path_label_list",
                    help="path of numy array with segmentation label values")
parser.add_argument("--save_label_list", type=str, default=None, dest="save_label_list",
                    help="path where to save the label list, if not initially provided")
parser.add_argument("--n_neutral_labels", type=int, default=1, dest="n_neutral_labels",
                    help="number of non-sided labels, default is 1")
parser.add_argument("--batchsize", type=int, dest="batchsize", default=1, help="batch size")
parser.add_argument("--target_res", type=str, dest="target_res", default=None,
                    help="path to numpy array with target resolution for the segmentation maps")
parser.add_argument("--unet_shape", type=str, dest="output_shape", default=None, help="size of unet's inputs")

# Augmentation parameters
parser.add_argument("--no_flip", action='store_false', dest="flipping", help="deactivate flipping")
parser.add_argument("--flip_rl_only", action='store_true', dest="flip_rl_only", help="only flip along right/left axis")
parser.add_argument("--scaling", type=infer, default=.15, dest="scaling_bounds", help="scaling range")
parser.add_argument("--rotation", type=infer, default=15, dest="rotation_bounds", help="rotation range")
parser.add_argument("--90_rotations", action='store_true', dest="enable_90_rotations",
                    help="wehther to introduce additional rotations of 0, 90, 180, or 270 degrees.")
parser.add_argument("--shearing", type=infer, default=.012, dest="shearing_bounds", help="shearing range")
parser.add_argument("--translation", type=infer, default=False, dest="translation_bounds", help="translation range")
parser.add_argument("--nonlin_std", type=float, default=3, dest="nonlin_std",
                    help="std dev. of the elastic deformation before upsampling to image size")
parser.add_argument("--nonlin_shape_factor", type=float, default=.04, dest="nonlin_shape_factor",
                    help="ratio between the size of the image and the sampled elastic deformation")
parser.add_argument("--bias_field_std", type=float, default=.5, dest="bias_field_std",
                    help="std dev. of the bias field before upsampling to image size")
parser.add_argument("--bias_shape_factor", type=float, default=.025, dest="bias_shape_factor",
                    help="ratio between the size of the image and the sampled bias field")
parser.add_argument("--same_bias_for_all_channels", action='store_true', dest="same_bias_for_all_channels",
                    help="whether to apply the same bias field to all channels.")
parser.add_argument("--no_intensity_augmentation", action='store_false', dest="augment_intensitites",
                    help="deactivate intensity augmentation")
parser.add_argument("--noise_std", type=float, default=1., dest="noise_std", help="std dev. of the gaussian noise")
parser.add_argument("--augment_channels_together", action='store_false', dest="augment_channels_separately",
                    help="augment intensities of all channels together rather than separately.")

# Architecture parameters
parser.add_argument("--n_levels", type=int, dest="n_levels", default=3, help="number of levels for the UNet")
parser.add_argument("--conv_per_level", type=int, dest="nb_conv_per_level", default=2, help="conv layers par level")
parser.add_argument("--conv_size", type=int, dest="conv_size", default=3, help="size of unet's convolution masks")
parser.add_argument("--unet_features", type=int, dest="unet_feat_count", default=24, help="features of the first layer")
parser.add_argument("--feat_mult", type=int, dest="feat_multiplier", default=2,
                    help="number by which to multiply the number of features at each level")
parser.add_argument("--dropout", type=float, dest="dropout", default=0, help="dropout probability")
parser.add_argument("--activation", type=str, dest="activation", default='elu', help="activation function")

# training parameters
parser.add_argument("--lr", type=float, dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--lr_decay", type=float, dest="lr_decay", default=1e-6, help="learning rate decay")
parser.add_argument("--wl2_epochs", type=int, dest="wl2_epochs", default=5, help="number of iterations")
parser.add_argument("--dice_epochs", type=int, dest="dice_epochs", default=200, help="number of iterations")
parser.add_argument("--steps_per_epoch", type=int, dest="steps_per_epoch", default=1000,
                    help="frequency of model saves")
parser.add_argument("--checkpoint", type=str, dest="checkpoint", default=None, help="h5 model to resume training.")

args = parser.parse_args()
training(**vars(args))
