import unittest
import numpy as np
# import IGTD
import tac.utils

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

    def test_jsd(self):
        """
        Test jensen_shannon_distance
        """
        X = np.array([[1,2,3],[4,5,6],[7,8,9]])
        X = IGTD.utils.normalized_data(X)
        assert np.array_equal(IGTD.utils.jensen_shannon_distance(X=X),\
                            np.array([[0., 0.10693419, 0.14038336],
                                      [0.10693419, 0.,0.02301584],
                                      [0.14038336, 0.02301584, 0.]]))
if __name__ == '__main__':
    # unittest.main()
    print(sys.path)