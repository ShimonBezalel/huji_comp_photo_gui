from .image_utils import *
import numpy as np
from memoized import memoized
from skimage.transform.pyramids import pyramid_gaussian
from skimage.transform import rescale, resize
import pickle

STEREO_MIN = -1
STEREO_MAX = 1
MOVE_MIN = -1
MOVE_MAX = 1
SHIFT_MIN = 0
SHIFT_MAX = 1

DEBUG = True
EPSILON = 0.001


class Gui:

    def __init__(self):
        self._series = None
        self._rows, self._cols, self._channels, self._frames = None, None, None, None
        self._gray_scale_series = None
        self._pyramid = None
        self._gray_scale_pyramid = None
        self._last_result = None
        self._gui_live_result_height = None
        self._gui_live_result_width = None
        self._motion_vec = None
        self._avg_motion = None
        self._interp_move = None
        self._interp_stereo = None
        self._interp_shift = None

    def setup(self, series_path, suffix="", extension="", height=500, width=900):
        """

        :param extension:
        :param suffix:
        :param series_path:
        :param height:
        :param width:
        """
        self._series = open_series(series_path, suffix=suffix, extension=extension)
        self._rows, self._cols, self._channels, self._frames = self._series.shape
        self._gray_scale_series = rgb2gray_series(self._series)
        self._pyramid = pyramid_gaussian(self._series, downscale=2, multichannel=True)
        self._gray_scale_pyramid = pyramid_gaussian(self._gray_scale_series, downscale=2, multichannel=True)
        self._last_result = self._series[..., 0]
        self._gui_live_result_height = height
        self._gui_live_result_width = width

        if DEBUG:
            if os.path.exists("test.pkl"):
                with open('test.pkl', 'rb') as f:
                    self._motion_vec = pickle.load(f)
            else:
                with open('test.pkl', 'wb') as f:
                    self._motion_vec = calculate_motion3(self._series)
                    pickle.dump(self._motion_vec, f)
        else:
            self._motion_vec = calculate_motion3(self._series)
        self._avg_motion = np.mean(self._motion_vec)
        self._interp_move = interp1d((-1, 1), (-(np.pi / 2), np.pi / 2))
        self._interp_stereo = interp1d((-1, 1), (0, self._cols - 1))
        self._interp_shift = interp1d((0, 1), (0, self._frames - 1))

    def _resize_result(self):
        h, w, c = self._last_result.shape
        if h > w:
            proportional_width = int((self._gui_live_result_width / h) * w)
            resize_shape = (self._gui_live_result_height, proportional_width, 3)
            padding = (self._gui_live_result_width - proportional_width) // 2
            padding_widths = ((0, 0), (padding, padding), (0, 0))
        elif w > h:
            proportional_height = int((self._gui_live_result_height / w) * h)
            resize_shape = (proportional_height, self._gui_live_result_width, 3)
            padding = (self._gui_live_result_height - proportional_height) // 2
            padding_widths = ((padding, padding), (0, 0), (0, 0))
        else:  # w == h
            if self._gui_live_result_width > self._gui_live_result_height:
                resize_shape = (self._gui_live_result_height, self._gui_live_result_height, 3)
                padding = (self._gui_live_result_width - self._gui_live_result_height) // 2
                padding_widths = ((0, 0), (padding, padding), (0, 0))
            else:
                resize_shape = (self._gui_live_result_width, self._gui_live_result_width, 3)
                padding = (self._gui_live_result_height - self._gui_live_result_width) // 2
                padding_widths = ((padding, padding), (0, 0), (0, 0))

        resized = resize(self._last_result, output_shape=resize_shape)
        output = np.pad(resized, pad_width=padding_widths, mode='constant')
        return output

    def _calc_slice(self, move, stereo, shift=0.5):
        """

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
            y0 = min(max(y(0), 0), self._cols - 1)
            x0 = x(y0)
            yn = min(max(y(self._frames - 1), 0), self._cols - 1)
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

    def focus(self, depth):
        self._last_result = focus(self._series, depth=depth, motion_vectors=self._motion_vec)
        return self._resize_result()

    def get_last_result(self, resized=True):
        if resized:
            return self._resize_result()
        return self._last_result

    def viewpoint(self, slice=None, move=None, stereo=None, shift=None):
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
            slice = self._calc_slice(move=move, stereo=stereo, shift=shift)
        else:
            assert np.array(slice).shape == (2, 2)
            (f_start, c_start), (f_end, c_end) = slice
            for frame in (f_start, f_end):
                assert (0 <= frame < self._frames)
            for col in (c_start, c_end):
                assert (0 <= col < self._cols)

        r = stitch(self._series, slice=slice, avg_motion=self._avg_motion)
        self._last_result = (r - r.min()) / (np.ptp(r))
        return self._resize_result()

