import numpy as np
from memoized import memoized
from skimage.transform.pyramids import pyramid_gaussian
from skimage.transform import rescale

def stitch(im_series, slice, compact_series=None, factor=1):
    """
    Stitch together columns from a given image series along the slice
         ___________*________
        |          /        |
        |         /         |
        |        /          |
        |_______/___________|
               *
    :param im_series: n X m X {1,3} X k array (k frames)
    :param slice: (first frame, first column) , (last frame, last col)
    :return:
    """
    factor = 1 if compact_series is None else factor
    (f_start, c_start), (f_end, c_end) = slice
    assert f_start <= f_end
    if f_start == f_end:
        return im_series[..., f_start]
    relevant_frames = im_series[..., f_start: f_end]
    result_image = np.zeros_like(relevant_frames[..., 0])  #todo: What is the result shape? Same as frame 1
    number_of_frames = f_end - f_start
    # The follow method uses a trick of rescaling the images for images between frames
    cols = np.linspace(c_start, c_end, number_of_frames * factor)
    cols_whole = np.round(cols).astype(np.int)
    # compact_series = rescale(im_series, (1, 1, 1, factor))
    compact_series = compact_series if compact_series is not None else  im_series
    relevant_cols = np.swapaxes(compact_series[::, cols_whole, ::, np.arange(number_of_frames * factor)], 0, 1)
    result_image[::, cols_whole, ::] = relevant_cols
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
    #return cropped area from result image
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
    grayscale_series = np.squeeze(im_series)   # shape =  n X m X k
    motion_vectors = np.zeros(shape=(k-1, 2))
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

