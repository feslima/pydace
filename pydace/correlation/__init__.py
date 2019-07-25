import numpy as np


def corr(theta, d, correlation='corrgauss', jacobian=False):
    """ Correlation model implementation.

    See page 19 of DACE.pdf explaining on how new equations must be added.

    Parameters
    ----------
    theta : (1, N) or (N, 1) ndarray
        Parameters in the correlation function. The number of elements in `theta` must be equal to the dimension N given
        in `d`. Also, a scalar value is allowed (this corresponds to an isotropic model: all :math:`\\theta_{j}` values
        are the same.
    d : (M,N) ndarray
        M-by-N array containing the differences between design sites.
    correlation: {'corrgauss'}, optional
        Correlation model type. Only Gaussian correlation model implemented.
    jacobian: bool, optional
        Whether or not to compute the jacobian. Be aware that setting this option to True the returned output will be a
        tuple with two elements. Default value is False.

    Returns
    -------
    r : ndarray
        Correlations.
    dr : (M, N) ndarray, optional
        M-by-N array containing the Jacobian information of r. Only calculated when the parameter `jacobian` is set to
        'yes'.

    Raises
    ------
    ValueError
        When the number of elements of `theta` is not n.
        When a invalid `jacobian` option is specified.
    NotImplementedError
        When `correlation` is not a valid option.


    """
    # TODO: implement other correlation models
    if type(jacobian) is not bool:
        raise ValueError(
            "Invalid jacobian option. Valid values are True and False.")

    m, n = d.shape  # number of differences and dimension of data

    if correlation == 'corrgauss':
        if theta.size == 1:
            theta = np.tile(theta, (1, n))
        elif theta.size != n:
            raise ValueError(f'Length of theta must be 1 or {n}.')

        # original -- td = d ** 2 * np.tile(-theta.reshape(-1, 1).T, (m, 1))
        td = d * -theta.flatten()
        # keepdims=True is to force r to be a matrix column
        r = np.exp(np.sum(d * td, axis=1, keepdims=False))

        if jacobian:
            # original -- dr = np.tile(-2 * theta.reshape(-1, 1).T, (m, 1)) * d * np.tile(r, (1, n))
            dr = 2 * td * r[:, np.newaxis]

    else:
        raise NotImplementedError('Correlation not implemented or invalid.')

    if jacobian:
        return r, dr
    else:
        return r
