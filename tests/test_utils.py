import unittest
import tac.utils
import numpy as np


class TestUtils(unittest.TestCase):

    def test_pad(self):
        """
        Test pad function.
        """
        assert np.array_equal(tac.utils.pad([1,1], 2), [0, 1, 1, 0])
        assert np.array_equal(tac.utils.pad([1], 2), [0, 0, 1, 0])
        assert np.array_equal(tac.utils.pad([1], 3), [0, 0, 0, 0, 1, 0, 0, 0, 0])
        assert np.array_equal(tac.utils.pad([1, 2, 3, 4, 5], 3, mode='zeros'), [0, 0, 1, 2, 3, 4, 5, 0, 0])

    def test_nearest_odd_root(self):
        """
        Test nearest_odd function.
        """
        assert tac.utils.nearest_odd_root(25) ==  5
        assert tac.utils.nearest_odd_root(26) == 7
        assert tac.utils.nearest_odd_root(48) == 7


if __name__ == '__main__':
    unittest.main()