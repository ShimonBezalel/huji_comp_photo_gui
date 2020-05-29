from unittest import TestCase
from backend.alg.image_utils import stitch, open_series
import numpy as np
from matplotlib import pyplot as plt
import os
import time

class TestStitch(TestCase):
    def test_stitch_artificial(self):
        for s in [(20,30,3,40), (200,300,3,100)]:
            p = 1
            for i in s:
                p *= i
            series = np.linspace(0, 1, p).reshape(s)
            red_ball = np.zeros(shape=s[:-1])
            ball_red = s[0] // 8
            red_ball[s[0]//2:s[0]//2 + ball_red, 0:ball_red, 0] = 1
            series_ball = np.copy(series)
            for i in range(s[-1]):
                series_ball[..., i] += np.roll(red_ball, i, axis=1)

            slice1 = ((0,0),(s[-1]-1, s[1]-1))
            res = stitch(series, slice1)
            plt.imshow(res)
            plt.show()

            slice2 = ((2,s[0]//8),(s[-1]-2,s[0]-s[0]//8))
            res2 = stitch(series_ball, slice2)
            plt.imshow(res2)
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

        i = 0
        t = time.time()
        for col in range(0, m, 4):
            slice3 = ((0, col), (k - 1, col))
            res = stitch(im_series, slice3)
            i += 1
        time.time() - t

        print("Avg Time: {}".format((time.time() - t) / i))
        plt.imshow(res)
        plt.show()




