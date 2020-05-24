import os
import numpy as np
from matplotlib import pyplot as plt
from backend.alg.image_utils import stitch
import time

from skimage.transform import rescale

if __name__ == '__main__':
    example_path = "/Users/shimonheimowitz/PycharmProjects/huji_comp_photo_gui/sample_data/apples"
    suffix = "APPLE"
    for root, dirs, files in os.walk(example_path):
        k = len(files)
        t = time.time()
        all_images = [plt.imread(os.path.join(example_path, "{}{:03d}.jpg".format(suffix, i)), format="jpg")
                      for i in range(1, k+1)]
        print("Read {} images in {} s".format(k, time.time() - t))
        im_series = np.stack(all_images, axis=-1)
        downscale = 0.8
        se = rescale(im_series, (downscale, downscale, 1, 1))
        factor = 1
        t = time.time()
        if factor != 1:
            padded = rescale(se, (1, 1, 1, factor))
            print("Scaled up in by {} in {} s".format(factor, time.time() - t))
        else:
            padded = se
        s = ((10, 350 * downscale), (200, 10))
        t = time.time()
        res = stitch(se, s, compact_series=padded, factor=factor)
        print("Solved stitching in {} s".format(time.time() - t))

        plt.imshow(res)
        plt.show()
