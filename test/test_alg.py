from unittest import TestCase
from backend.alg.image_utils import *
import numpy as np
from matplotlib import pyplot as plt
import os
import time
from skimage.transform.pyramids import pyramid_gaussian
import pickle


def gen_artificial_series(s, with_ball=True, smooth=True, brighten=True, step=None):
    p = 1
    for i in s:
        p *= i
    method = np.random.uniform if not smooth else np.linspace
    series = method(0, 1, p).reshape(s)
    if brighten:
        series *= np.linspace(0, 1, s[-1])[np.newaxis, np.newaxis, np.newaxis, ::]
    if not with_ball:
        return series

    red_ball = np.zeros(shape=s[:-1])
    ball_radius = s[0] // 6
    red_ball[s[0] // 2:s[0] // 2 + ball_radius, 0:ball_radius, 0] = 1
    series_ball = np.copy(series)
    m = min(s[1], s[-1])
    if step:
        for i, j in zip(np.arange(m), np.arange(0, m, step)):
            series_ball[..., i] += np.roll(red_ball, j, axis=1)

    else:
        for i, j in zip(np.arange(m),
                        np.linspace(0, s[1] - ball_radius - 1, m).astype(np.int)):
            series_ball[..., i] += np.roll(red_ball, j, axis=1)

    return series_ball

class TestGrayScale(TestCase):
    def test_rgb2gray_series(self):
        for s in [(20,30,3,40), (200,300,3,100)]:
            series_ball = gen_artificial_series(s)

            for i in range(0, s[-1], s[-1]//4):
                plt.imshow(series_ball[..., i])
                plt.show()

            gray_series = rgb2gray_series(series_ball)
            for i in range(0, s[-1],  s[-1]//4):
                plt.imshow(gray_series[..., i], cmap='gray')
                plt.show()


class TestFocus(TestCase):
    def test_inf_depth_artificial(self):
        for s in [(10, 20, 3, 25), (20, 40, 1, 50)]:
            for series in [
                gen_artificial_series(s, brighten=False),
                gen_artificial_series(s, brighten=False, step=1),
                gen_artificial_series(s, brighten=False, step=5)
            ]:
                res = focus(series, depth=1)
                print(res.shape)
                plt.imshow(res)
                plt.show()

    def test_inf_depth(self):
        example = "apple"
        suffix = example.upper()
        p = os.path.join("..", "sample_data", example)
        series = open_series(p, suffix=suffix, extension="jpg")
        res = focus(series, depth=1, motion_vectors=["dummy"])
        print(res.shape)
        plt.imshow(res)
        plt.show()


class TestMotion(TestCase):
    def test_calculate_motion_artificial(self):
        for s in [(10, 20, 3, 50), (100, 200, 1, 200)]:
            for series in [
                gen_artificial_series(s, brighten=False),
                gen_artificial_series(s, brighten=False, step=1),
                gen_artificial_series(s, brighten=False, step=5)
            ]:


                v1 = calculate_motion(series[..., 2:8])
                v2 = calculate_motion(series[..., 32:38])
                print(np.round(np.mean(v1-v2, axis=0), 3))

                plt.imshow(v1-v2)
                plt.show()

    def test_calculate_motion(self):
        example = "apple"
        suffix = example.upper()
        p = os.path.join("..", "sample_data", example)
        im_series = open_series(p, suffix=suffix, extension="jpg")
        v1 = calculate_motion(im_series[..., 2:8])
        print(np.round(v1, 3))

        v2 = calculate_motion(im_series[..., 32:38])
        print(np.round(v2, 3))

        print(np.round(np.mean(v1-v2, axis=0), 3))

        # plt.imshow(v1-v2)
        # plt.show()

        t = time.time()
        # v = calculate_motion(im_series)
        print("Time: {}".format(time.time() - t))
        # plt.imshow(v)
        # plt.show()

    def test_calculate_motion_time(self):
        example = "apple"
        suffix = example.upper()
        p = os.path.join("..", "sample_data", example)
        im_series = open_series(p, suffix=suffix, extension="jpg")
        gray = rgb2gray_series(im_series)
        pyramid = tuple(pyramid_gaussian(gray, multichannel=True, downscale=2))
        res = []
        for i in reversed(range(1, len(pyramid) - 3)):
            level = pyramid[i]
            t = time.time()
            v = calculate_motion(level[..., 10:20])
            tt = time.time() - t
            print(tt)
            res.append((i, v, level.shape, tt))

        print()
        for level, v, s, t in res:
            x = v * (2 ** level)
            print(level, np.mean(x, axis=0), s, t)


    def test_motion_full(self):
        example = "apple"
        suffix = example.upper()
        p = os.path.join("..", "sample_data", example)
        im_series = open_series(p, suffix=suffix, extension="jpg")
        v = calculate_motion(im_series[..., 0:100])
        print(v)

            # print(v * 2 ** i)
        # t = time.time()
        # v = calculate_motion(im_series)
        # print("Time: {}".format(time.time() - t))
        # plt.imshow(v)
        # plt.show()

    def test_motion_downscaled(self):
        example = "apple"
        suffix = example.upper()
        p = os.path.join("..", "sample_data", example)
        im_series = open_series(p, suffix=suffix, extension="jpg")
        f = calculate_motion3(im_series, 16)
        for i in range(0, f.shape[-1], 10):
            plt.imshow(np.linalg.norm(f[..., i], axis=0))
            plt.show()

        with open("flow_apple", 'wb') as pf:
            pickle.dump(f, pf)


class TestStitch(TestCase):
    def test_estimate_factor(self):
        width = 400
        for delta_cols, delta_frames in [
            (400, 200), (200, 200), (200, 400), (2, 100), (100, 2), (2, 400), (400, 2)
        ]:
            print(delta_cols, delta_frames, estimate_factor(width, delta_frames, delta_cols))
            print(-delta_cols, delta_frames, estimate_factor(width, delta_frames, -delta_cols))



    def test_stitch_artificial(self):
        for s in [(20,30,3,40), (200,300,3,100)]:
            n, m , c, k = s
            series_ball = gen_artificial_series(s)


            slice1 = ((0,0),(k-1, m-1))
            res = stitch(series_ball, slice1)
            plt.imshow(res)
            plt.show()

            slice2 = ((2,m//8),(k-2,m-m//8))
            res2 = stitch(series_ball, slice2)
            plt.imshow(res2)
            plt.show()

            slice3 = ((2,10),(k-2,10))
            res3 = stitch(series_ball, slice3)
            plt.imshow(res3)
            plt.show()

    def test_stitch_image_file(self):
        example = "apple"
        suffix = example.upper()
        p = os.path.join("..", "sample_data", example)
        im_series = open_series(p, suffix=suffix, extension="jpg")

        (n, m, c, k) = im_series.shape
        print("Res: {}".format(im_series.shape))

        slice1 = ((0, 0), (k - 1, m - 1))
        t = time.time()
        res = stitch(im_series, slice1)
        print("Time: {}".format(time.time() - t))
        plt.imshow(res)
        plt.show()

        slice2 = ((3, m // 8), (k - 3, m - (m // 8)))
        t = time.time()
        res = stitch(im_series, slice2)
        print("Time: {}".format(time.time() - t))
        plt.imshow(res)
        plt.show()

        slice3 = ((3, m - (m // 8)), (k - 3, m // 8))
        t = time.time()
        res = stitch(im_series, slice3)
        print("Time: {}".format(time.time() - t))
        plt.imshow(res)
        plt.show()

        slice4 = ((3, 10), (k - 3, 10))
        t = time.time()
        res = stitch(im_series, slice4)
        print("Time: {}".format(time.time() - t))
        plt.imshow(res)
        plt.show()


        i = 0
        t = time.time()
        for col in range(0, m, 10):
            slice3 = ((0, col), (k - 1, col))
            res = stitch(im_series, slice3, 0.6)
            plt.imshow(res)
            plt.show()
            i += 1
        time.time() - t

        print("Avg Time: {}".format((time.time() - t) / i))
        plt.imshow(res)
        plt.show()




