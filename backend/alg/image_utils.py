import numpy as np
from memoized import memoized
from skimage.transform.pyramids import pyramid_gaussian
from skimage.transform import rescale, resize
import os
import time
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from skimage.registration import optical_flow_tvl1
from skimage.color import rgb2gray

import cv2
from scipy.ndimage.interpolation import shift
from skimage.util.shape import view_as_windows

FACTOR_MAX = 10
FACTOR_MIN = 1

DEBUG = True



def open_series(path, suffix="", extension="jpg", zero_index=False):
    for root, dirs, files in os.walk(path):
        k = len(files)
        file_format = "{}{:03d}.{}" if k > 100 else "{}{:02d}.{}"
        t = time.time()
        first, last = (0, k) if zero_index else (1, k+1)
        all_images = [plt.imread(os.path.join(path, file_format.format(suffix, i, extension)), format=extension)
                      for i in range(first, last)]
        print("Read {} images in {} s".format(k, time.time() - t))
        im_series = np.stack(all_images, axis=-1)
        im_series = im_series / np.max(im_series)
        return im_series


@memoized
def estimate_factor(m, delta_frames, delta_cols):


    # (_, c_start), (_, c_end) = slice
    # m = shape[1]
    # delta = c_end - c_start

    factorize = interp1d((0, (m / 2) + 1), (FACTOR_MIN, FACTOR_MAX))

    f = factorize(np.abs(delta_cols / delta_frames))
    if delta_cols < 0:
        f = 1/f
    return f


def stitch(im_series, slice, avg_motion=3):
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

    if f_start == f_end:
        return im_series[..., f_start]
    assert f_start < f_end
    factor = estimate_factor(im_series.shape[1], f_end - f_start, c_end - c_start)  # todo: What is the result shape? Same as frame 1
    slice_width = np.abs(np.round(avg_motion * factor))
    number_of_frames = f_end - f_start
    number_of_columns = c_end - c_start

    input_image_shape = im_series[..., 0].shape
    # result_shape = (input_image_shape[0], int(max(number_of_frames, number_of_columns) * factor), input_image_shape[2])
    result_shape = (input_image_shape[0], int(number_of_frames * slice_width), input_image_shape[2])
    result_image = np.zeros(shape=result_shape)
    cols = np.linspace(c_start, c_end, result_shape[1])  # ie [ 2.3, 2.4, 2.9, 3.2 , ...]
    cols_right = np.ceil(cols).astype(np.int)  # ie [ 3, 3, 3, 4 , ...]
    cols_right_weights = cols - cols_right  # ie [ 0.3, 0.4, 0.9, 0.2 , ...]

    cols_left = np.floor(cols).astype(np.int) # ie [ 2, 2, 2, 3 , ...]
    cols_left_weights = 1 - cols_right_weights  # ie [ 0.7, 0.6, 0.1, 0.8 , ...]

    frame_indices = np.repeat(np.arange(f_start, f_end)[..., np.newaxis], slice_width, axis=-1).flatten()

    pixels_right = im_series[::, cols_right, ::, frame_indices]
    pixels_left = im_series[::, cols_left.reshape(frame_indices.shape), ::, frame_indices]
    result_image[::, np.arange(result_shape[1]), ::] = np.swapaxes(
        pixels_right * cols_right_weights[..., np.newaxis, np.newaxis] +
        pixels_left * cols_left_weights[..., np.newaxis, np.newaxis], 0, 1)
    return result_image


def focus(im_series, shift_factor, depth=None):
	"""

	:param im_series:
	:param depth:
	:return:
	"""
	p = np.full((im_series.shape[0], im_series.shape[1] - 1, im_series.shape[2], im_series.shape[3]), 0)
	a_ext = np.concatenate((p, im_series, p), axis=1)
	y, x, c, s = im_series.shape
	v = view_as_windows(a_ext, (y, x, c, 1))
	shift_vec = np.linspace(start=(-np.floor(s / 2)) * shift_factor, stop=(np.ceil(s / 2) - 1) * shift_facter, num=s)
	ser = v[0, shift_vec.astype(int) - x, 0, np.arange(s)]
	shifted_series = np.transpose(np.squeeze(ser), (1, 2, 3, 0))
	im = np.median(shifted_series, axis=-1)
	return im


def rgb2gray_series(series):
    """
    Coverts an image series to gray
    :param series:
    :return:
    """
    s = np.swapaxes(series, 2, 3)
    as_gray = rgb2gray(s)
    return as_gray


def calculate_motion(im_series):
    """
    Returns a list of vectors of motion between each image. For k images, we will get k-1 2D vectors
    :param im_series: n X m X {1, 3} X k Array - k frames of n X m images, where the images can be grayscale or 3 channels
    :return: Array of shape (k-1, 2)
    """
    if len(im_series.shape) == 3:
        m, n, k = im_series.shape
        gray_series = im_series

    elif len(im_series.shape) == 4:
        m, n, c, k = im_series.shape
        if c == 3:
            gray_series = rgb2gray_series(im_series)  # shape =  n X m X k
        elif c == 1:
            gray_series = np.squeeze(im_series)
        else:
            raise ValueError("Mis-aligned series shape {}. Must have 1 or 3 channels".format(im_series.shape))

    else:
        raise ValueError("Wrong shape for image series {}".format(im_series.shape))

    assert k >= 2, "Motion is computed for at least 2 images, received {}.".format(k)

    pyramid = tuple(pyramid_gaussian(gray_series, multichannel=True, downscale=2))
    relevant_levels = list(reversed(range(len(pyramid) - 4)))

    interim_results = []
    for level in relevant_levels:
        series : np.ndarray = pyramid[level]
        motion_vectors = np.zeros(shape=(k - 1, 2))
        base_im = series[..., 0]
        for frame_i in range(k - 1):
            moved_im = series[..., frame_i + 1]
            # im1, im2 = series[..., frame_i], series[..., frame_i + 1]
            flow = optical_flow_tvl1(base_im, moved_im)  # returns shape 2, m, n

            # range for bins edges [[x_min, x_max], y_min, y_max]]
            limit_range = np.array([np.min(flow, axis=(1, 2)), np.max(flow, axis=(1, 2))]).transpose()
            if DEBUG:
                plt.hist2d(flow[0].flatten(), flow[1].flatten(), bins=10, range=limit_range)
                plt.show()
                continue

            hist, x_edges, y_edges = np.histogram2d(flow[0].flatten(), flow[1].flatten(), bins=100, range=limit_range)
            i, j = np.unravel_index(np.argmax(hist), hist.shape)

            x_val = (x_edges[i] + x_edges[i+1]) / 2  # mid-range in bin
            y_val = (y_edges[j] + y_edges[j+1]) / 2  # mid-range in bin

            motion_vectors[frame_i] = [x_val, y_val]
        normalized_vec = motion_vectors * (2 ** level)
        if interim_results:
            prev_results = interim_results[-1]
            m_prev, m_new = np.mean(prev_results, axis=0), np.mean(normalized_vec, axis=0)
            print(np.abs(m_new - m_prev))
            if almost_equal(m_prev, m_new):
                return motion_vectors

        interim_results.append(normalized_vec)
    return interim_results[-1]


def calculate_motion2(im_series):
    """
    Returns a list of vectors of motion between each image. For k images, we will get k-1 2D vectors
    :param im_series: n X m X {1, 3} X k Array - k frames of n X m images, where the images can be grayscale or 3 channels
    :return: Array of shape (k-1, 2)
    """
    if len(im_series.shape) == 3:
        m, n, k = im_series.shape
        gray_series = im_series

    elif len(im_series.shape) == 4:
        m, n, c, k = im_series.shape
        if c == 3:
            gray_series = rgb2gray_series(im_series)  # shape =  n X m X k
        elif c == 1:
            gray_series = np.squeeze(im_series)
        else:
            raise ValueError("Mis-aligned series shape {}. Must have 1 or 3 channels".format(im_series.shape))

    else:
        raise ValueError("Wrong shape for image series {}".format(im_series.shape))

    assert k >= 2, "Motion is computed for at least 2 images, received {}.".format(k)

    pyramid = tuple(pyramid_gaussian(gray_series, multichannel=True, downscale=2))
    relevant_levels = list(reversed(range(len(pyramid) - 4)))
    print(relevant_levels)

    interim_results = []
    r = []
    for level in relevant_levels:
        print("LEVEL {}".format(level))
        series : np.ndarray = pyramid[level]
        base_im = series[..., 0]
        medians = np.zeros((2, k-1))
        flows = np.zeros((2, ) + base_im.shape + (k-1,))
        for frame_i in range(1, k):
            other_im = series[..., frame_i]
            flow = optical_flow_tvl1(base_im, other_im)  # returns shape 2, m, n
            flow *= (2 ** level)  # normalize to base resolution
            m = np.median(flow, axis=(1, 2))
            medians[..., frame_i-1] = m
            flows[..., frame_i-1] = flow
        if interim_results:
            m_prev = interim_results[-1]
            a_m, a_m_p = np.average(medians, axis=1), np.average(m_prev, axis=1)
            print(a_m, a_m_p, np.abs(a_m - a_m_p))
            if almost_equal(a_m, a_m_p):
                r.append(resize(flows, output_shape=(2,) + tuple(np.array(gray_series.shape) - [0,0,1])))
        interim_results.append(medians)
    return r


def calculate_motion3(im_series, scale=4):
    """
    Returns a list of vectors of motion between each image. For k images, we will get k-1 2D vectors
    :param im_series: n X m X {1, 3} X k Array - k frames of n X m images, where the images can be grayscale or 3 channels
    :return: Array of shape (k-1, 2)
    """
    if len(im_series.shape) == 3:
        m, n, k = im_series.shape
        gray_series = im_series

    elif len(im_series.shape) == 4:
        m, n, c, k = im_series.shape
        if c == 3:
            gray_series = rgb2gray_series(im_series)  # shape =  n X m X k
        elif c == 1:
            gray_series = np.squeeze(im_series)
        else:
            raise ValueError("Mis-aligned series shape {}. Must have 1 or 3 channels".format(im_series.shape))

    else:
        raise ValueError("Wrong shape for image series {}".format(im_series.shape))

    assert k >= 2, "Motion is computed for at least 2 images, received {}.".format(k)

    downscaled_series = rescale(gray_series, multichannel=True, scale=1/scale)
    low_res_flows = np.zeros((2, ) + downscaled_series[..., 0].shape + (k-1,))
    for frame_i in range(0, k-1):
        first_im, second_im = downscaled_series[..., frame_i], downscaled_series[..., frame_i + 1]
        flow = optical_flow_tvl1(first_im, second_im)  # returns shape 2, m, n
        flow *= scale  # normalize to original resolution
        low_res_flows[..., frame_i] = flow
    original_flow_shape = (2, ) + gray_series.shape[: -1] + (gray_series.shape[-1] - 1,)
    original_res_flows = resize(low_res_flows, output_shape=original_flow_shape)
    return original_res_flows


def almost_equal(v1, v2, tolerance=0.1):
    d = np.abs(v1 - v2)
    return np.all(d < tolerance)


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


def strided_indexing_roll(a, r):
	p = np.full((a.shape[0], a.shape[1] - 1, a.shape[2], a.shape[3]), 0)
	a_ext = np.concatenate((p, a, p), axis=1)
	y, x, c, s = a.shape
	v = view_as_windows(a_ext, (y, x, c, 1))
	shift_vec = np.linspace(start=(-np.floor(s / 2)) * r, stop=(np.ceil(s / 2) - 1) * r, num=s)
	ser = v[0, np.round(shift_vec).astype(np.int) - x, 0, np.arange(s)]
	return np.transpose(np.squeeze(ser), (1, 2, 3, 0))
