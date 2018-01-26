import numpy as np
from numpy.testing import assert_almost_equal

from sklearn.cross_decomposition import CCA

from context import meegkit  # noqa
from meegkit.utils import demean, nt_cca, tscov


# x=randn(1000,11);
# y=randn(1000,9);
# x=nt_demean(x);
# y=nt_demean(y);
# [A1,B1,R1]=canoncorr(x,y);
# [A2,B2,R2]=nt_cca(x,y);
# A2=A2*sqrt(size(x,1));
# B2=B2*sqrt(size(y,1));

# figure(1); clf;
# subplot 211; plot([R1' R2']);
# if mean(A1(:).*A2(:))<0; A2=-A2; end


def test_cca():
    """Test CCA."""
    x = np.random.rand(1000, 11)
    y = np.random.rand(1000, 9)
    x = demean(x).squeeze()
    y = demean(y).squeeze()

    [A, B, R] = nt_cca(x, y)  # if mean(A1(:).*A2(:))<0; A2=-A2; end
    X1 = np.dot(x, A)
    Y1 = np.dot(y, B)

    cca = CCA(n_components=9)
    X2, Y2 = cca.fit_transform(x, y)

    C1 = np.cov(np.hstack((X1, Y1)))
    C2 = np.cov(np.hstack((X2, Y2)))
    # import matplotlib.pyplot as plt
    # f, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.imshow(C1)
    # ax2.imshow(C2)
    # plt.show()
    # assert_almost_equal(C1, C2, decimal=4)

    # Compare with Matlab's canoncorr
    x = np.array([[16, 2, 3, 13],
                  [5, 11, 10, 8],
                  [9, 7, 6, 12],
                  [4, 14, 15, 1]])
    y = np.array([[1, 5, 9],
                  [2, 6, 10],
                  [3, 7, 11],
                  [4, 8, 12]])
    [A, B, R] = nt_cca(x, y)
    canoncorr = np.array([[-0.5000, -0.6708],
                          [-0.5000, 0.2236],
                          [-0.5000, -0.2236],
                          [-0.5000, 0.6708]])

    # close enough
    assert_almost_equal(np.cov(np.dot(x, A).T), np.cov(canoncorr.T), decimal=4)

if __name__ == '__main__':
    import nose
    nose.run(defaultTest=__name__)
