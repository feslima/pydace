import itertools

import numpy as np
from pyDOE2 import lhs


def lhsdesign(n, min_range, max_range, k=5, include_vertices=False):
    """Returns the Latin Hypercube Sampling for a given range of values.

    Parameters
    ----------
    n : int
        Number of samples of the hypercube.
    min_range : np.array
        1-by-p or p-by-1 array containing the minimum values for each variable.
    max_range : np.array
        1-by-p or p-by-1 array containing the maximum values for each variable.
    k : int, optional
        Number of iterations to attempt to improve the design.
    include_vertices : bool
        To include or not the vertices of the hypercube in the sample.

    Returns
    -------
    out : np.array
        n-by-p array containing the Latin Hypercube Sampling.

    Raises
    ------
    ValueError
        If ndim of either `min_range` or `max_range` is not 2.

        If the `min_range` or `max_range` aren't vectors.


    """

    # check input ranges dimensions. If ndim != 1, raise error
    if min_range.ndim != 1 or max_range.ndim != 1:
        raise ValueError("Input ranges must be 1D arrays.")
    else:
        # both have ndim == 1, check if they have the same size
        if min_range.size != max_range.size:
            raise ValueError("min_range and max_range must have the same number of elements")

        # min_range = min_range.reshape(1, -1)
        # max_range = max_range.reshape(1, -1)

        p = min_range.size

    # proceed with normal calculations
    slope = np.tile(max_range - min_range, (n, 1))
    offset = np.tile(min_range, (n, 1))

    # create normalized LH
    x_normalized = lhs(p, samples=n, iterations=k, criterion='maximin')

    if include_vertices:
        vertices = get_vertices(min_range, max_range)

        # scale and return the LH
        return np.vstack((x_normalized * slope + offset, vertices))
    else:
        # scale and return the LH
        return x_normalized * slope + offset


def get_vertices_index(n):
    return np.asarray(list(itertools.product('01', repeat=n))).astype(np.int)


def get_vertices(lb, ub):
    """ Returns all vertices of the hypercube.

    Parameters
    ----------
    lb : np.array
        Lower bound of the hypercube.
    ub : np.array
        Upper bound of the hypercube.

    Returns
    -------
    out : np.array
        2**n-by-n array containing all the vertices of the n-dimensional hypercube. Each line corresponds to a vertex
        of the hypercube

    Raises
    ------
    ValueError
        if the bounds ´lb´ or ´ub´ does not have same shape or number of elements in each

    """
    if lb.ndim == 1 or ub.ndim == 1:
        lb = lb.reshape(1, -1)
        ub = ub.reshape(1, -1)

    if lb.shape != ub.shape:
        raise ValueError(f"´lb´ and ´ub´ must have the same number of elements. ´lb´ = {lb.size} and ´ub´= {ub.size}")

    n = lb.shape[1]  # number of dimensions
    m = 2 ** n  # number of vertices
    vertices_index = get_vertices_index(n)

    cat_placeholder = np.vstack((lb, ub))

    vertices = np.zeros(vertices_index.shape)
    for j in np.arange(n):
        for i in np.arange(m):
            vertices[i, j] = cat_placeholder[vertices_index[i, j], j]

    return vertices