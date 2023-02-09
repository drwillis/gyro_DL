"""
 RALIGN - Rigid alignment of two sets of points in k-dimensional
          Euclidean space.  Given two sets of points in
          correspondence, this function computes the scaling,
          rotation, and translation that define the transform TR
          that minimizes the sum of squared errors between TR(X)
          and its corresponding points in Y.  This routine takes
          O(n k^3)-time.

 Inputs:
   X - a k x n matrix whose columns are points
   Y - a k x n matrix whose columns are points that correspond to
       the points in X
 Outputs:
   c, R, t - the scaling, rotation matrix, and translation vector
             defining the linear map TR as

                       TR(x) = c * R * x + t

             such that the average norm of TR(X(:, i) - Y(:, i))
             is minimized.
"""

"""
Copyright: Carlo Nicolini, 2013
Code adapted from the Mark Paskin Matlab version
from http://openslam.informatik.uni-freiburg.de/data/svn/tjtf/trunk/matlab/ralign.m 
"""

import numpy as np

def ralign(X, Y):
    m, n = X.shape

    mx = X.mean(1)
    my = Y.mean(1)
    Xc = X - np.tile(mx, (n, 1)).T
    Yc = Y - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(Xc * Xc, 0))
    sy = np.mean(np.sum(Yc * Yc, 0))

    Sxy = np.dot(Yc, Xc.T) / n

    U, D, V = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = V.T.copy()
    # print U,"\n\n",D,"\n\n",V
    r = np.linalg.matrix_rank(Sxy)
    d = np.linalg.det(Sxy)
    S = np.eye(m)
    if r >= (m - 1):
        if (np.linalg.det(Sxy) < 0):
            S[m-1, m-1] = -1;
        else:
            R = np.eye(m)
            c = 1
            t = my - mx
            return R, c, t

    R = np.dot(np.dot(U, S), V.T)
    c = np.trace(np.dot(np.diag(D), S)) / sx
    t = my - c * np.dot(R, mx)
    return R, c, t


"""
## References
- [Umeyama's paper](http://edge.cs.drexel.edu/Dmitriy/Matching_and_Metrics/Umeyama/um.pdf)
- [CarloNicolini's python implementation](https://gist.github.com/CarloNicolini/7118015)
"""
def similarity_transform(from_points, to_points):
    assert len(from_points.shape) == 2, "from_points must be a m x n array"
    assert from_points.shape == to_points.shape, "from_points and to_points must have the same shape"

    N, m = from_points.shape

    mean_from = from_points.mean(axis=0)
    mean_to = to_points.mean(axis=0)

    delta_from = from_points - mean_from  # N x m
    delta_to = to_points - mean_to  # N x m

    sigma_from = (delta_from * delta_from).sum(axis=1).mean()
    sigma_to = (delta_to * delta_to).sum(axis=1).mean()

    cov_matrix = delta_to.T.dot(delta_from) / N

    U, d, V_t = np.linalg.svd(cov_matrix, full_matrices=True)
    cov_rank = np.linalg.matrix_rank(cov_matrix)
    S = np.eye(m)

    if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
        S[m - 1, m - 1] = -1
    elif cov_rank < m - 1:
        # raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
        R = np.eye(m)
        c = 1
        t = mean_to - mean_from
        return R, c, t

    R = U.dot(S).dot(V_t)
    c = (d * S.diagonal()).sum() / sigma_from
    t = mean_to - c * R.dot(mean_from)
    return R, c, t


# Run an example test
# We have 3 points in 3D. Every point is a column vector of this matrix A
# A = np.array([[0.57215, 0.37512, 0.37551], [0.23318, 0.86846, 0.98642], [0.79969, 0.96778, 0.27493]])
# # Deep copy A to get B
# B = A.copy()
# # and sum a translation on z axis (3rd row) of 10 units
# B[2, :] = B[2, :] + 10
#
# # Reconstruct the transformation with ralign.ralign
# R, c, t = ralign(A, B)
# print("Rotation matrix=\n", R, "\nScaling coefficient=", c, "\nTranslation vector=", t)