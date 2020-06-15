from unittest import TestCase
from backend.alg.image_api import Gui, interp1d
import os
import numpy as np
from matplotlib import pyplot as plt

class TestGuiClass(TestCase):

    def setUp(self):
        self._mock_gui_class = Gui()

        # example = "apple"
        # suffix = example.upper()
        # p = os.path.join("..", "sample_data", example)
        #
        # self._mock_gui_class.setup(series_path=p, suffix=suffix, extension="jpg")

        self._mock_gui_class._gui_live_result_width = 900
        self._mock_gui_class._gui_live_result_height = 500


    def test_resize(self):
        for shape in [
            (300, 400, 3), (400, 300, 3), (355, 355, 3), (300, 400, 1),
            (100, 900, 3), (900, 100, 3), (100, 100, 3), (10, 1000, 1)
                      ]:
            self._mock_gui_class._last_result = np.ones(shape)
            res = self._mock_gui_class._resize_result()
            plt.imshow(res)
            plt.show()

    def test_slice(self):
        for s in [(300, 400), (1000, 10)]:
            self._mock_gui_class._cols, self._mock_gui_class._frames = s
            self._mock_gui_class._interp_move = interp1d((-1, 1), (-(np.pi / 2), np.pi / 2))
            self._mock_gui_class._interp_stereo = interp1d((-1, 1), (0, self._mock_gui_class._cols - 1))
            self._mock_gui_class._interp_shift = interp1d((0, 1), (0, self._mock_gui_class._frames - 1))

            for move in np.linspace(-0.99, 0.99, 10):
                for stereo in np.linspace(-0.99, 0.99, 10):
                    for shift in np.linspace(0.01, 0.99, 10):
                        print(move, stereo, shift)
                        sl = self._mock_gui_class.params_to_slice(move, stereo, shift)
                        print("\t", sl)
                        print()
                        for pair in sl:
                            self.assertGreater(self._mock_gui_class._frames, pair[0])
                            self.assertGreaterEqual(pair[0], 0)
                            self.assertGreater(self._mock_gui_class._cols, pair[1])
                            self.assertGreaterEqual(pair[1], 0)
                        self.assertGreaterEqual(sl[1][0], sl[0][0], "first frame must be lesser")

                    print("----------")
                print("============")
