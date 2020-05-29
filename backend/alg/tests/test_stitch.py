from unittest import TestCase
from ..image_utils import stitch
import numpy as np
from matplotlib import pyplot as plt

class TestStitch(TestCase):
    def test_stitch(self):
        for s in [(20,30,3,40), (200,300,3,100)]:
            p = 1
            for i in s:
                p *= i
            series = np.linspace(0, 1, p).reshape(s)
            red_ball = np.zeros(shape=s[:-1])
            red_ball[s[0]//2:s[0]//2 + 3, 0:3, 0] = 1
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