# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents).
All Rights Reserved. Permission to use, copy, modify, and distribute this
software and its documentation for educational, research, and not-for-profit
purposes, without fee and without a signed licensing agreement, is hereby
granted, provided that the above copyright notice, this paragraph and the
following two paragraphs appear in all copies, modifications, and
distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150
Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201,
otl@berkeley.edu,
http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Simple utility functions.

Authors
-------
Jeff Mahler, Vishal Satish, Lucas Manuelli
"""
from functools import reduce
import os
import sys

import numpy as np
import skimage.transform as skt

from autolab_core import Logger
from .enums import GripperMode
from PIL import Image, ImageDraw

# Set up logger.
logger = Logger.get_logger("gqcnn/utils/utils.py")


def is_py2():
    return sys.version[0] == "2"


def set_cuda_visible_devices(gpu_list):
    """Sets CUDA_VISIBLE_DEVICES environment variable to only show certain
    gpus.

    Note
    ----
    If gpu_list is empty does nothing.

    Parameters
    ----------
    gpu_list : list
        List of gpus to set as visible.
    """
    if len(gpu_list) == 0:
        return

    cuda_visible_devices = ""
    for gpu in gpu_list:
        cuda_visible_devices += str(gpu) + ","

    logger.info(
        "Setting CUDA_VISIBLE_DEVICES = {}".format(cuda_visible_devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def pose_dim(gripper_mode):
    """Returns the dimensions of the pose vector for the given
    gripper mode.

    Parameters
    ----------
    gripper_mode: :obj:`GripperMode`
        Enum for gripper mode, see optimizer_constants.py for all possible
        gripper modes.

    Returns
    -------
    :obj:`numpy.ndarray`
        Sliced pose_data corresponding to gripper mode.
    """
    if gripper_mode == GripperMode.PARALLEL_JAW:
        return 1
    elif gripper_mode == GripperMode.LEGACY_PARALLEL_JAW:
        return 1

    else:
        raise ValueError(
            "Gripper mode '{}' not supported.".format(gripper_mode))


def read_pose_data(pose_arr, gripper_mode, pose_input):
    """Read the pose data and slice it according to the specified gripper mode.

    Parameters
    ----------
    pose_arr: :obj:`numpy.ndarray`
        Full pose data array read in from file.
    gripper_mode: :obj:`GripperMode`
        Enum for gripper mode, see optimizer_constants.py for all possible
        gripper modes.
    pose_input: :dict:
        Dictionary of which explicit inputs to use in the pose stream.

    Returns
    -------
    :obj:`numpy.ndarray`
        Sliced pose_data corresponding to input data mode.
    """
    if pose_input is not None:
        values = np.array(())
        if pose_arr.ndim == 1:
            if pose_input['x'] == 1:
                values = np.r_[values, pose_arr[0]]
            if pose_input['y'] == 1:
                values = np.vstack((values, pose_arr[1]))
            if pose_input['x'] == 0 and pose_input['z'] == 1:
                values = np.r_[values, pose_arr[2]]
            elif pose_input['z'] == 1:
                values = np.vstack((values, pose_arr[2]))
            if pose_input['quaternion'] != 0:
                for i in range(4, 8):
                    values = np.vstack((values, pose_arr[i]))
        else:
            if pose_input['x'] == 1:
                values = np.r_[values, pose_arr[:, 0]]
            if pose_input['y'] == 1:
                values = np.vstack((values, pose_arr[:, 1]))
            if pose_input['x'] == 0 and pose_input['z'] == 1:
                values = np.r_[values, pose_arr[:, 2]]
            elif pose_input['z'] == 1:
                values = np.vstack((values, pose_arr[:, 2]))
            if pose_input['quaternion'] != 0:
                for i in range(4, 8):
                    values = np.vstack((values, pose_arr[:, i]))
        return values.T
    else:
        if gripper_mode == GripperMode.PARALLEL_JAW:
            if pose_arr.ndim == 1:
                return pose_arr[2:3]
            else:
                return pose_arr[:, 2:3]
        elif gripper_mode == GripperMode.LEGACY_PARALLEL_JAW:
            if pose_arr.ndim == 1:
                return pose_arr[2:3]
            else:
                return pose_arr[:, 2:3]
        else:
            raise ValueError(
                "Gripper mode '{}' not supported.".format(gripper_mode))


def reduce_shape(shape):
    """Get shape of a layer for flattening."""
    shape = [x.value for x in shape[1:]]
    f = lambda x, y: 1 if y is None else x * y  # noqa: E731
    return reduce(f, shape, 1)


def weight_name_to_layer_name(weight_name):
    """Convert the name of weights to the layer name."""
    tokens = weight_name.split("_")
    type_name = tokens[-1]

    # Modern naming convention.
    if type_name == "weights" or type_name == "bias":
        if len(tokens) >= 3 and tokens[-3] == "input":
            return weight_name[:weight_name.rfind("input") - 1]
        return weight_name[:weight_name.rfind(type_name) - 1]
    # Legacy.
    if type_name == "im":
        return weight_name[:-4]
    if type_name == "pose":
        return weight_name[:-6]
    return weight_name[:-1]


def imresize(image, size, interp="nearest"):
    """Wrapper over `skimage.transform.resize` to mimic `scipy.misc.imresize`.
    Copied from https://github.com/BerkeleyAutomation/perception/blob/master/perception/image.py#L38.  # noqa: E501

    Since `scipy.misc.imresize` has been removed in version 1.3.*, instead use
    `skimage.transform.resize`. The "lanczos" and "cubic" interpolation methods
    are not supported by `skimage.transform.resize`, however there is now
    "biquadratic", "biquartic", and "biquintic".

    Parameters
    ----------
    image : :obj:`numpy.ndarray`
        The image to resize.

    size : int, float, or tuple
        * int   - Percentage of current size.
        * float - Fraction of current size.
        * tuple - Size of the output image.

    interp : :obj:`str`, optional
        Interpolation to use for re-sizing ("neartest", "bilinear",
        "biquadratic", "bicubic", "biquartic", "biquintic"). Default is
        "nearest".

    Returns
    -------
    :obj:`np.ndarray`
        The resized image.
    """
    skt_interp_map = {
        "nearest": 0,
        "bilinear": 1,
        "biquadratic": 2,
        "bicubic": 3,
        "biquartic": 4,
        "biquintic": 5
    }
    if interp in ("lanczos", "cubic"):
        raise ValueError("'lanczos' and 'cubic'"
                         " interpolation are no longer supported.")
    assert interp in skt_interp_map, ("Interpolation '{}' not"
                                      " supported.".format(interp))

    if isinstance(size, (tuple, list)):
        output_shape = size
    elif isinstance(size, (float)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size
        output_shape = tuple(np_shape.astype(int))
    elif isinstance(size, (int)):
        np_shape = np.asarray(image.shape).astype(np.float32)
        np_shape[0:2] *= size / 100.0
        output_shape = tuple(np_shape.astype(int))
    else:
        raise ValueError("Invalid type for size '{}'.".format(type(size)))

    return skt.resize(image,
                      output_shape,
                      order=skt_interp_map[interp],
                      anti_aliasing=False,
                      mode="constant")


def scale_data(data, scaling='max', sc_min=0.6, sc_max=0.8):
    """Scales a numpy array to [0, 255].

        Parameters
        ----------
        data: :obj:`numpy.ndarray`
            Data to be scaled.
        scaling: str
            Scaling method. Can be 'fixed' to scale between fixed values,
            or 'max' to scale between the minimum and maximum of the data.
            Defaults to 'max'.
        sc_min: float
            Lower bound for fixed scaling. Defaults to 0.6.
        sc_max: float
            Upper bound for fixed scaling. Defaults to 0.8

        Returns
        -------
        :obj:`numpy.ndarray`
            Scaled numpy array with the same shape as input array data.
    """
    data_fl = data.flatten()
    if scaling == 'fixed':
        scaled = np.interp(data_fl, (sc_min, sc_max), (0, 255), left=0, right=255)
    elif scaling == 'max':
        scaled = np.interp(data_fl, (min(data_fl), max(data_fl)), (0, 255), left=0, right=255)
    else:
        raise AttributeError
    integ = scaled.astype(np.uint8)
    integ.resize(data.shape)
    return integ


def centre_rotate_crop_resize(image, grasp_x, grasp_y, angle):
    """Centres and crops an image to grasp centre. Resizes and crops to 32x32 pixel.

            Parameters
            ----------
            image: :obj:`numpy.ndarray`
                Depth image to be cropped.
            grasp_x: int
                Offset in x of the grasp centre to image centre. [px]
            grasp_y: int
                Offset in y of the grasp centre to image centre. [px]
            angle: float
                Grasp roll for image rotation

            Returns
            -------
            :obj:`numpy.ndarray`
                Cropped and rotated 32x32 pixel depth image.
        """
    r0 = grasp_x - 75
    r1 = grasp_x + 75
    r2 = grasp_y - 75
    r3 = grasp_y + 75

    im = Image.fromarray(image).crop((r0, r2, r1, r3)).rotate(np.rad2deg(angle)).resize((50, 50),
                                                                                        resample=Image.BILINEAR)

    return np.array(im)[9:41, 9:41]


def random_im_crop(image, final_size, grasp_x, grasp_y, kappa=0, debug=False):
    """Crops an image with normally distributed offset to grasp centre.

            Parameters
            ----------
            image: :obj:`numpy.ndarray`
                Depth image to be cropped.
            final_size: tuple
                Size the image should have after cropping.
            grasp_x: int
                Offset in x of the grasp centre to image centre. [px]
            grasp_y: int
                Offset in y of the grasp centre to image centre. [px]
            kappa: int
                Maximum x/y offset of the grasp centre in the new image. Defaults to 0 [px]
            debug: bool
                Boolean to debug by visualising the image + grasp before and after cropping. Default: False.

            Returns
            -------
            :obj:`numpy.ndarray`
                Cropped depth image.
            tuple
                New grasp centre in (x,y) [px].
        """
    # Get centre of image
    (cx, cy) = image.squeeze().shape

    # Get half width of resulting image
    d_x = final_size[0] // 2
    d_y = final_size[1] // 2

    if grasp_x is None:
        grasp_x = cx // 2
    if grasp_y is None:
        grasp_y = cy // 2

    if debug:
        orig_im = Image.fromarray(scale_data(image)).convert('RGB')
        draw_im = ImageDraw.Draw(orig_im)
        draw_im.line([grasp_x, grasp_y, grasp_x, grasp_y], fill=(255, 0, 0, 255))
        orig_im.save('/home/anna/Desktop/orig.png')

    # Get bounds so that grasp is visible within the new image
    bound_x = (grasp_x - d_x, grasp_x + d_x)
    bound_y = (grasp_y - d_y, grasp_y + d_y)

    # Sample image centre based on grasp position from a uniform distribution
    _x = np.random.uniform(low=grasp_x + kappa, high=grasp_x - kappa)
    _y = np.random.uniform(low=grasp_y + kappa, high=grasp_y - kappa)

    centre_x = int(min(max(round(_x), bound_x[0]), bound_x[1]))
    centre_y = int(min(max(round(_y), bound_y[0]), bound_y[1]))

    # Get image bounds based on new image centre
    r0 = centre_x - d_x
    r1 = centre_x + d_x
    r2 = centre_y - d_y
    r3 = centre_y + d_y

    # Sampling bounds exceed original image
    if 0 > r0 or r0 > cx or 0 > r1 or r1 > cx or 0 > r2 or r2 > cy or 0 > r3 or r3 > cy:
        return None, (0, 0)

    new_image = image[r2: r3, r0: r1]
    new_grasp_x = grasp_x - centre_x
    new_grasp_y = grasp_y - centre_y

    if debug:
        new_im = Image.fromarray(scale_data(new_image)).convert('RGB')
        draw_im = ImageDraw.Draw(new_im)
        draw_im.line([d_x + new_grasp_x, d_y + new_grasp_y,
                      d_x + new_grasp_x, d_x + new_grasp_y],
                     fill=(255, 0, 0, 255))
        new_im.resize((300, 300)).save('/home/anna/Desktop/new.png')

    return new_image, (new_grasp_x, new_grasp_y)