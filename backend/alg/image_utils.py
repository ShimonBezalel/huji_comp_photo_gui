import numpy as np
from memoized import memoized
from skimage.transform.pyramids import pyramid_gaussian
from skimage.transform import rescale
import os
import time
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

FACTOR_MAX = 2
FACTOR_MIN = 0.7


def open_series(path, suffix="", extension="jpg"):
    for root, dirs, files in os.walk(path):
        k = len(files)
        file_format = "{}{:03d}.{}" if k > 100 else "{}{:02d}.{}"
        t = time.time()
        all_images = [plt.imread(os.path.join(path, file_format.format(suffix, i, extension)), format=extension)
                      for i in range(1, k+1)]
        print("Read {} images in {} s".format(k, time.time() - t))
        im_series = np.stack(all_images, axis=-1)
        im_series = im_series / np.max(im_series)
        return im_series


@memoized
def estimate_factor(m, delta):


    # (_, c_start), (_, c_end) = slice
    # m = shape[1]
    # delta = c_end - c_start

    factorize = interp1d((-(m-1), m-1), (FACTOR_MIN, FACTOR_MAX))

    return factorize(delta)



def stitch(im_series, slice, factor=1):
    """
    Stitch together columns from a given image series along the slice
         ___________*________
        |          /        |
        |         /         |
        |        /          |
        |_______/___________|
               *
    :rtype: np.ndarray 2D image
    :param factor:
    :return:
    :param im_series: n X m X {1,3} X k array (k frames)
    :param slice: (first frame, first column) , (last frame, last col)
    :return:
    """
    (f_start, c_start), (f_end, c_end) = slice
    factor = estimate_factor(im_series.shape[1], c_end - c_start)  # todo: What is the result shape? Same as frame 1
    assert f_start <= f_end
    if f_start == f_end:
        return im_series[..., f_start]
    number_of_frames = f_end - f_start
    number_of_columns = c_end - c_start

    input_image_shape = im_series[..., 0].shape
    result_shape = (input_image_shape[0], int(max(number_of_frames, number_of_columns) * factor), input_image_shape[2])
    result_image = np.zeros(shape=result_shape)
    # The follow method uses a trick of rescaling the images for images between frames
    cols = np.linspace(c_start, c_end, result_shape[1])  # ie [ 2.3, 2.4, 2.9, 3.2 , ...]

    cols_right = np.floor(cols).astype(np.int)  # ie [ 2, 2, 2, 3 , ...]
    cols_right_weights = cols - cols_right  # ie [ 0.3, 0.4, 0.9, 0.2 , ...]

    cols_left = np.ceil(cols).astype(np.int)  # ie [ 3, 3, 3, 4 , ...]
    cols_left_weights = 1 - cols_right_weights  # ie [ 0.7, 0.6, 0.1, 0.8 , ...]

    pixels_right = im_series[::, cols_right, ::, np.round(
            np.linspace(f_start, f_end, result_shape[1])).astype(np.int)]
    pixels_left = im_series[::, cols_left, ::, np.round(
            np.linspace(f_start, f_end, result_shape[1])).astype(np.int)]
    result_image[::, np.arange(result_shape[1]), ::] = np.swapaxes(
        pixels_right * cols_right_weights[..., np.newaxis, np.newaxis] +
        pixels_left * cols_left_weights[..., np.newaxis, np.newaxis], 0, 1)
    return result_image


def refocus(im_series, depth):
    """

    :param im_series:
    :param depth:
    :return:
    """
    # calculate motion between images
    # estimate overall size of the expected result image
    # bring images to be overlapping by padding and moving by the motion vector
    # move images slightly using depth parameter
    # return cropped area from result image
    raise NotImplemented


def calculate_motion(im_series: np.ndarray):
    """
    Returns a list of vectors of motion between each image. For k images, we will get k-1 2D vectors
    :param im_series: n X m X {1, 3} X k Array - k frames of n X m images, where the images can be grayscale or 3 channels
    :return: Array of shape (k-1, 2)
    """
    m, n, c, k = im_series.shape
    assert k >= 2, "Motion is computed for at least 2 images, received {}.".format(k)
    # if c == 3:
    # reduce to grayscale images
    grayscale_series = np.squeeze(im_series)  # shape =  n X m X k
    motion_vectors = np.zeros(shape=(k - 1, 2))
    for i in range(k):
        im1, im2 = np.squeeze()
        motion_vectors[i] = estimate_motion(im1, im2)
    raise NotImplemented


def estimate_motion(im1: np.ndarray, im2: np.ndarray):
    """
    Receives two images 2D grayscale images of similar shape and estimates motion vector between them. Assume vector is
    2D with only translation and rotation.
    :param im1, im2: Images of shape n X m
    :return:
    """
    assert im1.shape == im2.shape, "Images must have the same shape. {} != {}".format(im1.shape, im2.shape)
    assert im1.shape.__len__() == 2, "Images must have shape nXm, got {}".format(im1.shape)
    # Build Laplacian (pyramid of downscaled images)
    # chache interim results (use the same images twice per caluclation)
    pyrm1, pyrm2 = gen_pyramid(im1), gen_pyramid(im1)

    # for each level in the pyramid:
    # Find optical flow using lucas canade

    # Assuming the transform is uniform in the image (no moving parts) we can vote for the best motion vector
    # Build (2D) histogram of vectors, wisely using the min/max as bucket ends.

    # Find vector with highest votes as motion vector.

    # if this vector matches last iteration's motion, you can return it as the best candidate

    # if finished iterating the pyramid, return last vector

    raise NotImplemented


@memoized
def gen_pyramid(im):
    """
    Wrapper for building pyramids with chached outputs.
    Builds a downscale pyramid (im.shape, im.shape/2, im.shape/4, ... , im.shape/(2**max_level)

    :param im:
    :param max_layer:
    :return:
    """
    return pyramid_gaussian(im, max_layer=10)
