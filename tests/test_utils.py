import numpy as np
from numpy.testing import assert_equal

from context import meegkit  # noqa
from meegkit.utils import multishift, shiftnd, shift, relshift, widen_mask


def test_multishift():
    """Test matrix multi-shifting."""
    # multishift()
    x = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]])
    assert_equal(multishift(x, [1], axis=0), np.array([[0, 0, 0, 0],
                                                       [1, 2, 3, 4]]))

    # relshift() 1d
    y, y_ref = relshift([1, 2, 3, 4], [11, 12, 13, 14], [1], axis=0)
    assert_equal(y.flatten(), [0, 1, 2, 3])
    assert_equal(y_ref.flatten(), [0, 12, 13, 14])

    y, y_ref = relshift(np.arange(1, 6), np.arange(1, 6), [0, 1], axis=0)
    assert_equal(y, np.array([[1., 0.],
                              [2., 1.],
                              [3., 2.],
                              [4., 3.],
                              [5., 4.]]))

    # relshift() 2d
    x = [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    x_ref = [[11, 12, 13, 14],
             [15, 16, 17, 18]]

    y, y_ref = relshift(x, x_ref, [0, 1])
    assert_equal(y[..., 0], [[1, 2, 3, 4],
                             [5, 6, 7, 8]])
    assert_equal(y[..., 1], [[0, 0, 0, 0],
                             [1, 2, 3, 4]])
    assert_equal(y_ref[..., 0], [[11, 12, 13, 14],
                                 [15, 16, 17, 18]])
    assert_equal(y_ref[..., 1], [[0, 0, 0, 0],
                                 [15, 16, 17, 18]])


def test_shift():
    """Test matrix shifting."""
    x = np.arange(10)

    assert_equal(shiftnd(x, 2), np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7]))

    # x2 = array([[0, 1, 2, 3, 4],
    #             [5, 6, 7, 8, 9]])
    x2 = np.reshape(x, (2, 5))

    # Test shifts along several dimensions and directions
    assert_equal(shiftnd(x2, 1), np.array([[0, 0, 1, 2, 3],
                                           [4, 5, 6, 7, 8]]))
    assert_equal(shiftnd(x2, -2), np.array([[2, 3, 4, 5, 6],
                                            [7, 8, 9, 0, 0]]))
    assert_equal(shiftnd(x2, 1, axis=0), np.array([[0, 0, 0, 0, 0],
                                                   [0, 1, 2, 3, 4]]))
    assert_equal(shiftnd(x2, -1, axis=0), np.array([[5, 6, 7, 8, 9],
                                                    [0, 0, 0, 0, 0]]))
    assert_equal(shiftnd(x2, 1, axis=1), np.array([[0, 0, 1, 2, 3],
                                                   [0, 5, 6, 7, 8]]))

    # Same for `shift`
    assert_equal(shift(x2, 1, axis=0), np.array([[0, 0, 0, 0, 0],
                                                 [0, 1, 2, 3, 4]]))
    assert_equal(shift(x2, -1, axis=0), np.array([[5, 6, 7, 8, 9],
                                                  [0, 0, 0, 0, 0]]))
    assert_equal(shift(x2, 1, axis=1), np.array([[0, 0, 1, 2, 3],
                                                 [0, 5, 6, 7, 8]]))


def test_widen_mask():
    """Test binary mask operations."""
    test = np.array([0, 0, 0, 1, 0, 0, 0])

    # test 1d
    assert_equal(widen_mask(test, -2), [0, 1, 1, 1, 0, 0, 0])
    assert_equal(widen_mask(test, 2), [0, 0, 0, 1, 1, 1, 0])

    # test nd
    assert_equal(widen_mask(test[None, :], -2, axis=1),
                 [[0, 1, 1, 1, 0, 0, 0], ])
    assert_equal(widen_mask(test[None, None, :], -2, axis=2),
                 [[[0, 1, 1, 1, 0, 0, 0], ], ])


if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
