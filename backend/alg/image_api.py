from .image_utils import *
import numpy as np
from skimage.transform import rescale, resize

import pickle

from skimage.transform import warp

STEREO_MIN = -1
STEREO_MAX = 1
MOVE_MIN = -1
MOVE_MAX = 1
SHIFT_MIN = 0
SHIFT_MAX = 1

DEPTH_FACTOR = 1

DEBUG = True
EPSILON = 0.001

EPSILON2 = 0.0001


def call_and_pickle(path, func, *args):
    if DEBUG:
        save_pickle = not os.path.exists(path)
        if not save_pickle:
            with open(path, 'rb') as f:
                res = pickle.load(f)
                save_pickle = res is None

        if save_pickle:
            with open(path, 'wb') as f:
                res = func(*args)
                pickle.dump(res, f)
    else:
        res = func(*args)
    return res

def wrap_motion(series, frames):
    flow = calculate_motion3(series)
    motion_vec = []
    for frame_i in range(0, frames - 1):
        motion_vec.append(np.median(flow[..., frame_i], axis=(1, 2)))
    motion_vec = np.array(motion_vec)
    return motion_vec


class Gui:

    def __init__(self, series_path, prefix="", extension="jpg", height=500, width=900, zero_index=False):
        """

		:param extension:
		:param prefix:
		:param series_path:
		:param height:
		:param width:
		:param zero_index:

		"""
        print("Opening image series...")
        self._series = open_series(series_path, prefix=prefix, extension=extension, zero_index=zero_index)
        if self._series is None:
            print("Series is empty")
            return
        self._rows, self._cols, self._channels, self._frames = self._series.shape
        self._gray_scale_series = rgb2gray_series(self._series)
        # self._pyramid = pyramid_gaussian(self._series, downscale=2, multichannel=True)
        # self._gray_scale_pyramid = pyramid_gaussian(self._gray_scale_series, downscale=2, multichannel=True)
        self._last_result = self._series[..., 0]
        self._gui_live_result_height = height
        self._gui_live_result_width = width
        print("Calculating motion...")
        # self._motion_flow = call_and_pickle('flow_{}.pkl'.format(prefix), calculate_motion3, self._series)
        # self._motion_vec = []
        # for frame_i in range(0, self._frames - 1):
        #     self._motion_vec.append(np.median(self._motion_flow[..., frame_i], axis=(1, 2)))
        # self._motion_vec = np.array(self._motion_vec)
        self._motion_vec = call_and_pickle('flow_{}.pkl'.format(prefix), wrap_motion, self._series, self._frames)
        print("Aligning vertically...")
        self._aligned = call_and_pickle('aligned_{}.pkl'.format(prefix), self._align_images)
        # self._aligned_row_cols = call_and_pickle('aligned_row_cols_{}.pkl'.format(suffix), self._align_images, True, True)
        self._avg_horizontal_motion = np.mean(self._motion_vec[..., 1])

        # input images motion goes right to left
        if self._avg_horizontal_motion > 0:
            self._series = np.flip(self._series, axis=-1)
            self._gray_scale_series = np.flip(self._gray_scale_series, axis=-1)
            self._motion_vec = np.flip(self._motion_vec, axis=-1) * np.array([1, -1])
            self._avg_horizontal_motion = np.mean(self._motion_vec[..., 1])

        self._interp_move = interp1d((-1, 1), (-(np.pi / 2), np.pi / 2))
        self._interp_stereo = interp1d((-1, 1), (self._cols - 1, 0))
        self._interp_shift = interp1d((0, 1), (0, self._frames - 1))

        self._interp_rev_move = interp1d((-(np.pi / 2), np.pi / 2), (-1, 1))
        self._interp_rev_stereo = interp1d((self._cols - 1, 0),     (-1, 1))
        self._interp_rev_shift = interp1d((0, self._frames - 1),    (0, 1))


    def _resize_result(self):
        """
        Resize the image in "last result" to fit the gui aspect ratio and pixel size, by padding with black.

        GUI: Aspect ratio A_g       Res: Aspect ratio A_r  =>    Resize

        Case 1: A_r > A_g

         -----------                 ---------------             -----------
        |           |               |               |           |-----------|
        |           |               |               |           |           |
        |           |                ---------------            |-----------|
         -----------                                             -----------

        Case 2: A_g > A_r

         -----------                 ---                         -----------
        |           |               |   |                       |   |   |   |
        |           |               |   |                       |   |   |   |
        |           |               |   |                       |   |   |   |
         -----------                 ---                         -----------

        Case 3: A_g = A_r
        Simply resize
        :return:
        """
        h, w, c = self._last_result.shape
        gui_aspect_ratio = self._gui_live_result_width / self._gui_live_result_height
        res_aspect_ratio = w / h

        # Case 1
        if gui_aspect_ratio < res_aspect_ratio:
            proportional_height = int((self._gui_live_result_width * h) / w)
            resize_shape = (proportional_height, self._gui_live_result_width, 3)
            padding = (self._gui_live_result_height - proportional_height) // 2
            padding_dims = ((padding, padding), (0, 0), (0, 0))
        # Case 2
        elif gui_aspect_ratio > res_aspect_ratio:
            proportional_width = int((self._gui_live_result_height * w) / h)
            resize_shape = (self._gui_live_result_height, proportional_width, 3)
            padding = (self._gui_live_result_width - proportional_width) // 2
            padding_dims = ((0, 0), (padding, padding), (0, 0))
        # Case 3
        else:  # gui_aspect_ratio == res_aspect_ratio
            resize_shape = (self._gui_live_result_height, self._gui_live_result_width, 3)
            padding_dims = ((0, 0), (0, 0), (0, 0))

        resized = resize(self._last_result, output_shape=resize_shape)
        output = np.pad(resized, pad_width=padding_dims, mode='constant')
        return output

    def focus(self, depth, center, radius):
        rows, columns, channels, frames = self._series.shape
        start_frame, end_frame = max(0, center - radius), min(columns, center + radius)
        shift_factor = depth
        # shift_factor = self.get_motion_vec()[start_frame:end_frame, 1].mean() * depth
        self._last_result = focus2(self._aligned[..., start_frame: end_frame],
                                   shift_vec=self._motion_vec[start_frame: end_frame, 1] * depth)
        # self._last_result = focus(self._aligned_row_cols[..., start_frame:end_frame], shift_factor=shift_factor)
        return self._resize_result()

    def slice_to_params(self, slice):
        (f_start, c_start), (f_end, c_end) = slice
        for f in [f_start, f_end]:
            assert 0 <= f <= self._frames, "Frames provided out of range"
        for c in [c_start, c_end]:
            assert 0 <= c <= self._cols, "Cols provided out of range"

        center_col = (c_start + c_end) / 2
        stereo = np.round(self._interp_rev_stereo(center_col), 3)

        center_frame = (f_start + f_end) / 2
        shift = np.round(self._interp_rev_shift(center_frame), 3)

        if f_start == f_end:
            if c_end >= c_start:
                move = 1
            else:
                move = -1
        else:
            delta_cols = c_end - c_start    #  may be negative
            delta_frames = f_end - f_start  #  asserted to not be 0 above
            angle_in_pixels = delta_cols / delta_frames
            angle_radians = np.arctan(angle_in_pixels)
            move = np.round(self._interp_rev_move(angle_radians), 3)

        return {'stereo': stereo, 'shift': shift, 'move': move}

    def params_to_slice(self, move, stereo, shift=0.5):
        """
        Converts continuous params into discreet pixel coordinates as slice.
		:param move:
		:param stereo:
		:return:
		"""
        angle = self._interp_move(move)
        center_frame = self._interp_shift(shift)
        # The angle doesnt match a line function so simply return intended frame as a slice
        # | Vertical
        if np.abs(np.abs(angle) - np.pi / 2) < EPSILON:
            # first to last col in the same frame
            frame = np.round(center_frame).astype(np.int)
            return (frame, 0), (frame, self._cols - 1)
        center_col = self._interp_stereo(stereo)

        # --- Horizontal
        if (np.abs(angle)) < EPSILON:
            # first to last frame, same col
            col = np.round(center_col).astype(np.int)
            return (0, col), (self._frames - 1, col)

        # y = mx + b
        angle_pixels = np.tan(angle)
        ratio = self._cols / self._frames
        y = lambda x: angle_pixels * (x - center_frame) + center_col
        x = lambda y: ((y - center_col) / angle_pixels) + center_frame
        """
		CASE 1:  Sharp angle

			  x = 0               x = frames-1
					   (xn, yn)
			____|________o__________|___    y = cols-1
				|       /           |
				|      /.           |
		Cols (X)|_____/_____________|___    y = 0
		^       |    o              |
		|        (x0, y0)              
		|       
		o----> frames (Y)

		"""

        if np.abs(angle_pixels) > ratio:
            x0 = min(max(x(0), 0), self._frames - 1)
            y0 = y(x0)
            xn = min(max(x(self._cols - 1), 0), self._frames - 1)
            yn = y(xn)
            res = (int(round(x0)), int(round(y0))), (int(round(xn)), int(round(yn)))

            # if x0 < 0:
            #     x0 = 0
            #     y0 = y(x0)
            # if xn < 0:
            #     xn = 0
            #     yn = y(xn)
            # if x0 > self._frames - 1:
            #     x0 = self._frames - 1
            #     y0 = y(x0)
            # if xn < 0:
            #     xn = 0
            #     yn = y(0)
            if angle_pixels < 0:
                res = tuple(reversed(res))
        else:
            """
			CASE 2:  Shallow angle

				  x = 0               x = frames-1

					____|___________________|___    y = cols-1
			  (x0, y0) o|``````````         |
						|       ````````````|o (xn, yn)
			Cols (X) ___|___________________|___    y = 0
			^           |                   |
			|                      
			|       
			o----> frames (Y)

			"""
            y0 = min(max(y(0), 0 + EPSILON2), self._cols - 1 - EPSILON2)
            x0 = x(y0)
            yn = min(max(y(self._frames - 1), 0 + EPSILON2), self._cols - 1 - EPSILON2)
            xn = x(yn)

            if x0 < 0:
                x0, y0 = 0, y(0)  # todo: should be y(0)
                xn, yn = self._frames - 1, y(self._frames - 1)
                if yn < 0:
                    xn = x(0)
                    yn = 0
                if yn > self._cols - 1:
                    xn = x(self._cols - 1)
                    yn = self._cols - 1

            elif xn > self._frames - 1:
                xn, yn = self._frames - 1, y(self._frames - 1)
                x0, y0 = 0, min(y(0), self._cols - 1)
                if y0 < 0:
                    x0 = x(0)
                    y0 = 0
                if y0 > self._cols - 1:
                    x0 = x(self._cols - 1)
                    y0 = self._cols - 1
                if yn < 0:
                    xn = x(0)
                    yn = 0
                if yn > self._cols - 1:
                    xn = x(self._cols - 1)
                    yn = self._cols - 1

            res = (int(round(x0)), int(round(y0))), (int(round(xn)), int(round(yn)))
        return res

    def get_last_result(self, resized=True):
        if resized:
            return self._resize_result()
        return self._last_result

    def viewpoint(self, slice=None, move=None, stereo=None, shift=None, factor=None):
        """

		:param slice:
		:param move:
		:param stereo:
		:return:
		"""
        if slice == None:
            assert (move != None) and (stereo != None), \
                "Must provide slice or move + stereo in function viewpoint"
            assert (MOVE_MIN <= move <= MOVE_MAX), \
                "move must both be between {} and {}, got ".format(MOVE_MIN, MOVE_MAX, move)

            assert (STEREO_MIN <= stereo <= STEREO_MAX), \
                "stereo must both be between {} and {}, got ".format(STEREO_MIN, STEREO_MAX, stereo)

            if shift is not None:
                assert (SHIFT_MIN <= shift <= SHIFT_MAX), \
                    "shift must both be between {} and {}, got ".format(SHIFT_MIN, SHIFT_MAX, shift)
            slice = self.params_to_slice(move=move, stereo=stereo, shift=shift)
            (f_start, c_start), (f_end, c_end) = slice
        else:
            assert np.array(slice).shape == (2, 2)
            (f_start, c_start), (f_end, c_end) = slice
            for frame in (f_start, f_end):
                assert 0 <= frame < self._frames, "Frames is out of range"
            for col in (c_start, c_end):
                assert 0 <= col < self._cols, "Cols is out of range"

        slice_width = None
        if not factor:
            delta_col = c_end - c_start
            delta_frame = (f_end - f_start) + EPSILON   # add epsilon to avoid zero division
            overall_motion = np.abs(np.sum(self._motion_vec[f_start:f_end, 1]))
            slice_width = min(max(1, (overall_motion + delta_col) / delta_frame), self._cols)

        r = stitch(self._aligned, slice=slice, avg_motion=self._avg_horizontal_motion, factor=factor, slice_width=slice_width)
        self._last_result = r  # (r - r.min()) / (np.ptp(r))
        return self._resize_result()

    def get_motion_vec(self):
        return self._motion_vec

    def get_motion_avg(self):
        return self._avg_horizontal_motion

    def _align_images(self, align_rows=True, align_cols=False):
        t = time.time()
        if align_rows or align_cols:
            print("Aligning {} {}".format("rows" if align_rows else "", "cols" if align_cols else ""))

            row_coords, col_coords, channel_coords, frame_coords = \
                np.meshgrid(np.arange(self._rows), np.arange(self._cols), np.arange(self._channels),
                            np.arange(self._frames - 1),
                            indexing='ij')
            vec_cum_sum = np.cumsum(self._motion_vec, axis=0)
            relative_vec = (vec_cum_sum - np.flip(vec_cum_sum, axis=0)) / 2

            if align_rows:
                row_coords = row_coords + relative_vec[..., 0]
            if align_cols:
                col_coords = col_coords + relative_vec[..., 1]

            coords = np.array((row_coords, col_coords, channel_coords, frame_coords))
            res = warp(self._series, coords)

        else:
            print("Nothing to align.".format())
            res = self._series.copy()

        print("Aligned images in {}".format(round(time.time() - t, 2)))
        return res
