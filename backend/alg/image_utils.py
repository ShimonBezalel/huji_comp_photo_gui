import numpy as np
from memoized import memoized
from skimage.transform.pyramids import pyramid_gaussian


def refocus(im_series, depth):
    """

    :param im_series:
    :param depth:
    :return:
    """
    pass

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
def gen_pyramid(im: np.ndarray, max_layer=10):
    """
    Wrapper for building pyramids with chached outputs.
    Builds a downscale pyramid (im.shape, im.shape/2, im.shape/4, ... , im.shape/(2**max_level)

    :param im:
    :param max_layer:
    :return:
    """
    return pyramid_gaussian(im, max_layer=max_layer)

