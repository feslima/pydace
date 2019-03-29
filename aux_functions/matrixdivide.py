import scipy as sp


def mldivide(A, b):
    """Solve systems of linear equations Ax = B for x.

    The matrices A and B must contain the same number of rows.
    If A is a square n-by-n matrix and B is a matrix with n rows,
    then x = A\B is a solution to the equation A*x = B, if it exists.

    Parameters
    ----------
    b : numpy.array
    A : numpy.array

    Returns
    -------
    x : numpy.array
        Solution of Ax = B

    Raises
    ------
    ValueError
        If the number of rows of A and B are not the same

    """

    if A.size == 1 and b.size == 1:
        # perform element wise division, since both of inputs are scalar
        return b / A

    # check if A is square
    m, n = A.shape

    if m != b.shape[0]:
        raise ValueError('The matrices A and b must have the same number of rows.')

    if m == n:  # A is indeed square
        # TODO: implement MATLAB mldivide algorithm
        return sp.linalg.solve(A, b)
    else:
        return sp.linalg.lstsq(A, b)[0]


def mrdivide(B, A):
    """Solve systems of linear equations xA = B for x.

    The matrices A and B must contain the same number of columns.
    If A is a square n-by-n matrix and B is a matrix with n columns,
    then x = B/A is a solution to the equation x*A = B, if it exists.

    Parameters
    ----------
    B : numpy.array
    A : numpy.array

    Returns
    -------
    x : numpy.array
        Solution of xA = B

    Raises
    ------
    ValueError
        If the number of columns of A and B are not the same

    """

    if (A.shape[0] == 1 and A.shape[1] == 1) or (B.shape[0] == 1 and B.shape[1] == 1):
        # perform element wise division, since one or both of inputs are scalar
        return B / A

    if B.shape[1] != A.shape[1]:
        raise ValueError('A and B must contain the same number of columns.')

    return mldivide(A.T, B.T).T
