import numpy as np


def regrpoly(S, polynomial='poly0', jacobian=False):
    """Returns the regression matrix (F) and its jacobian (dF) of a given polynomial.

    Parameters
    ----------
    S : numpy.array
        m-by-n array with design sites.
    polynomial : string
        String that defines the polynomial model (default 'poly0' - zero order).
        Valid options are 'poly1' and 'poly2' for first and second orders, respectively.
    jacobian: bool, optional
        Whether or not to compute the jacobian. Be aware that setting this option to True the returned output will be a
        tuple with two elements. Default value is False.

    Returns
    -------
    F: numpy.array
        Regression matrix (F) of the chosen polynomial.
            If `polynomial` == 'poly0', F is a m-by-1 matrix of ones.
            If `polynomial` == 'poly1', F is a m-by-(n+1) matrix.
            If `polynomial` == 'poly2', F is a m-by-(n+1)*(n+1)/2 matrix.
    dF: numpy.array, optional
        Jacobian (dF) of the chosen polynomial. Only calculated when the parameter `jacobian` is set to 'yes'.

    Raises
    ------
    ValueError
        If `polynomial` is an invalid option.
        If `jacobian` is an invalid option.

    """
    if type(jacobian) is not bool:
        raise ValueError("Invalid jacobian option. Valid values are True and False.")

    m, n = S.shape

    # normal info
    if polynomial == 'poly0':
        f = np.ones((m, 1))

    elif polynomial == 'poly1':
        f = np.hstack((np.ones((m, 1)), S))

    elif polynomial == 'poly2':
        nn = int((n + 1) * (n + 2) / 2)
        f = np.hstack((np.ones((m, 1)), S, np.zeros((m, nn - n - 1))))
        j = n + 1
        q = n

        for k in np.arange(1, n + 1):
            # the k:k+1 in S[:,k:k+1] is to extract the column as a Mx1 matrix
            f[:, j + np.arange(q)] = np.tile(S[:, (k - 1):k], (1, q)) * S[:, np.arange(k - 1, n)]

            j += q
            q -= 1

    else:
        raise ValueError('Invalid regression polynomial choice')

    # jacobian info
    if jacobian:
        if polynomial == 'poly0':
            df = np.zeros((n, 1))

        elif polynomial == 'poly1':
            df = np.hstack((np.zeros((n, 1)), np.eye(n)))

        elif polynomial == 'poly2':
            nn = int((n + 1) * (n + 2) / 2)
            df = np.hstack((np.zeros((n, 1)), np.eye(n), np.zeros((n, nn - n - 1))))
            j = n + 1
            q = n

            for k in np.arange(1, n + 1):
                # the k:k+1 in S[:,k:k+1] is to extract the column as a Mx1 matrix
                df[k - 1, j + np.arange(q)] = np.hstack((2 * S[0, (k - 1):k], S[0, np.arange(k, n)]))
                # np.arange(k, n) => k + 1:n

                for i in np.arange(1, n - k + 1):
                    df[k - 1 + i, j + i] = S[0, (k - 1):k]

                j += q
                q -= 1

        else:
            raise ValueError('Invalid regression polynomial choice')

        return f, df

    else:
        return f