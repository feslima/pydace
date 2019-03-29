import numpy as np
from .dacefit import regrpoly, corr
from aux_functions.matrixdivide import mldivide


def predictor(x, dmodel, compute_jacobian=False, compute_mse=False, compute_mse_jacobian=False):
    """
    Predictor for y(x) using the given DACE model. (Optimized version to enable fast calculations)

    Parameters
    ----------
    x : (M, N) ndarray
        Trial design sites with n dimensions.
    dmodel : DaceModel
        DACE model struct. See DaceModel class.
    compute_jacobian : bool, optional
        Whether or not to compute the jacobian.
    compute_mse : bool, optional
        Whether or not to compute the Mean Squared Error (MSE) of the prediction
    compute_mse_jacobian : bool, optional
        Whether or not to compute the MSE jacobian of the prediction.

    Returns
    -------
    y : (M,1) ndarray
        Predicted response at x.
    or1 : (M, ...) ndarray
        If m = 1, and `cjac` is set to 'yes', then `or1` is a gradient vector/Jacobian matrix of predictor. Otherwise,
        `or1` is a vector with m rows containing the estimated mean squared error of the predictor.
    or2 : float
        If m = 1, and `cmse` is set to 'yes', then `or2` is a the estimated mean squared error (MSE) of the predictor.
        Otherwise (m > 1 or `cmse' set to 'no'), it's a NaN type.
    dmse : (M, ...) ndarray
        The gradient vector/Jacobian Matrix of the MSE. Only available when m = 1 and `cmsejac` set to 'yes', otherwise
        it's a NaN type.

    """
    if x.ndim == 1:  # change from 1d array to 2d
        x = x[np.newaxis, :]

    if x.ndim > 2:
        raise ValueError("Input arrays of dimension higher than 2 aren't allowed.")

    if np.all(np.isnan(dmodel.beta)):
        raise ValueError("Kriging build is invalid because it contains NaN values.")

    m, n = dmodel.S.shape  # number of design sites and number of dimension
    sx = x.shape  # number of trial sites and their dimension

    if np.min(sx) == 1 and n > 1:  # single trial point
        nx = np.max(sx)

        if nx == n:
            mx = 1
            x = x.reshape(1, -1)  # row vector reshaping ´x(:).'´

    else:
        mx, nx = sx

    if nx != n:
        raise ValueError(f"Dimension of trial sites must be {n}.")

    # assign data
    dy, mse, dmse = np.NaN, np.NaN, np.NaN

    # normalize trial sites
    x = (x - dmodel.Ssc[[0], :]) / dmodel.Ssc[[1], :]
    q = dmodel.Ysc.shape[1]  # number of response functions

    if mx == 1:  # one site only
        dx = x - dmodel.S  # distance to design sites

        # get correlation a regression data depending whether or not jacobian info was required
        if compute_jacobian or compute_mse_jacobian:  # jacobian required
            f, df = regrpoly(x, polynomial=dmodel.regr, jacobian=True)
            r, dr = corr(dmodel.theta, dx, correlation=dmodel.corr, jacobian=True)

        else:
            f = regrpoly(x, polynomial=dmodel.regr)
            r = corr(dmodel.theta, dx, correlation=dmodel.corr)

        # compute the prediction
        # Scaled predictor
        sy = f @ dmodel.beta + (dmodel.gamma @ r).T

        # Predictor
        if q == 1:  # make sure the return is a scalar
            y = np.asscalar((dmodel.Ysc[[0], :] + dmodel.Ysc[[1], :] * sy).conj().T)
        else:  # otherwise, keep it as it is
            y = (dmodel.Ysc[[0], :] + dmodel.Ysc[[1], :] * sy).conj().T

        # compute the prediction jacobian
        if compute_jacobian:
            # scaled jacobian
            sdy = np.transpose(df @ dmodel.beta) + dmodel.gamma @ dr

            # unscaled jacobian
            dy = sdy * dmodel.Ysc[[1], :].T / dmodel.Ssc[[1], :]

            if q == 1:  # gradient as column vector for single dimension
                dy = dy.conj().T

        # compute MSE
        if compute_mse:
            # MSE
            rt = mldivide(dmodel.C, r)
            u = dmodel.Ft.T @ rt - f.T
            v = mldivide(dmodel.G, u)
            mse = np.tile(dmodel.sigma2, (mx, 1)) * np.tile(
                (1 + np.sum(v ** 2, axis=0) - np.sum(rt ** 2, axis=0)).conj().T,
                (1, q))

            if q == 1:  # make sure the return is a scalar if q == 1
                mse = np.asscalar(mse)

            # compute MSE jacobian
            if compute_mse:
                # scaled gradient as row vector
                Gv = mldivide(dmodel.G.conj().T, v)
                g = (dmodel.Ft @ Gv - rt).conj().T @ mldivide(dmodel.C, dr) - (df @ Gv).conj().T

                # unscaled MSE jacobian
                dmse = np.tile(2 * dmodel.sigma2.reshape(1, -1).conj().T, (1, nx)) * np.tile(g / dmodel.Ssc[[1], :], (q, 1))

                if q == 1:  # gradient as column vector for single dimension
                    dmse = dmse.conj().T

    else:  # several trial sites

        if compute_jacobian or compute_mse_jacobian:  # basic sanitation
            raise ValueError("Can't compute either prediction or MSE jacobian for several design sites.")

        # Get distance to design sites
        dx = np.zeros((mx * m, n))
        kk = np.arange(m).reshape(1, -1)

        for k in np.arange(mx):
            dx[kk, :] = x[k, :] - dmodel.S
            kk = kk + m

        # Get regression function or correlation
        f = regrpoly(x, polynomial=dmodel.regr, jacobian='no')
        r = np.reshape(corr(dmodel.theta, dx, correlation=dmodel.corr, jacobian='no'), (m, mx), order='F')

        # scaled predictor
        sy = f @ dmodel.beta + (dmodel.gamma @ r).T

        # predictor
        # org -- y = np.tile(dmodel.Ysc[[0], :], (mx, 1)) + np.tile(dmodel.Ysc[[1], :], (mx, 1)) * sy
        y = dmodel.Ysc[[0], :] + dmodel.Ysc[[1], :] * sy

        # MSE
        rt = mldivide(dmodel.C, r)
        u = mldivide(dmodel.G, dmodel.Ft.T @ rt - f.T)
        # org -- mse = np.tile(dmodel.sigma2, (mx, 1)) * np.tile((1 + colsum(u ** 2) - colsum(rt ** 2)).conj().T,(1, q))
        mse = dmodel.sigma2 * (1 + colsum(u ** 2) - colsum(rt ** 2)).conj().T

    return y, dy, mse, dmse


def predictor_legacy(x, dmodel):
    """
    WARNING: THIS IS THE ORIGINAL IMPLEMENTATION (NOT OPTIMIZED)
    Predictor for y(x) using the given DACE model

    Parameters
    ----------
    x : (M, N) ndarray
        Trial design sites with n dimensions.
    dmodel : DaceModel
        DACE model struct. See DaceModel class.

    Returns
    -------
    y : (M,1) ndarray
        Predicted response at x.
    or1 : (M, ...) ndarray
        If m = 1, then `or1` is a gradient vector/Jacobian matrix of predictor. Otherwise, `or1` is a vector with m rows
        containing the estimated mean squared error of the predictor.
    or2 : float
        If m = 1, then `or2` is a the estimated mean squared error (MSE) of the predictor. Otherwise, it's a NaN type.
    dmse : (M, ...) ndarray
        The gradient vector/Jacobian Matrix of the MSE. Only available when m = 1.

    """
    or1, or2, dmse = np.NaN, np.NaN, np.NaN  # default return value

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if np.all(np.isnan(dmodel.beta)):  # Adding 'all' check even though matlab's script doesn't
        y = np.NaN
        raise ValueError('Kriging builder not found.')

    m, n = dmodel.S.shape  # number of design sites and number of dimension
    sx = x.shape  # number of trial sites and their dimension

    if np.min(sx) == 1 and n > 1:  # single trial point
        nx = np.max(sx)

        if nx == n:
            mx = 1
            x = x.reshape(1, -1)  # row vector reshaping ´x(:).'´

    else:
        mx, nx = sx

    if nx != n:
        raise ValueError(f'Dimension of trial sites should be {n}')

    # normalize trial sites
    x = (x - np.tile(dmodel.Ssc[[0], :], (mx, 1))) / np.tile(dmodel.Ssc[[1], :], (mx, 1))
    q = dmodel.Ysc.shape[1]  # number of response functions
    y = np.zeros((mx, q))  # initialize result

    if mx == 1:  # one site only
        dx = np.tile(x, (m, 1)) - dmodel.S  # distance to design sites

        f, df = regrpoly(x,polynomial=dmodel.regr, jacobian='yes')
        r, dr = corr(dmodel.theta, dx, correlation=dmodel.corr, jacobian='yes')

        # scaled jacobian
        dy = np.transpose(df @ dmodel.beta) + dmodel.gamma @ dr

        # unscaled jacobian
        or1 = dy * dmodel.Ysc[[1], :].T / dmodel.Ssc[[1], :]

        # MSE
        rt = mldivide(dmodel.C, r)
        u = dmodel.Ft.T @ rt - f.T
        v = mldivide(dmodel.G, u)
        or2 = np.tile(dmodel.sigma2, (mx, 1)) * np.tile((1 + np.sum(v ** 2, axis=0) - np.sum(rt ** 2, axis=0)).conj().T,
                                                        (1, q))

        # gradient of MSE
        # scaled gradient as row vector
        Gv = mldivide(dmodel.G.conj().T, v)
        g = (dmodel.Ft @ Gv - rt).conj().T @ mldivide(dmodel.C, dr) - (df @ Gv).conj().T

        # unscaled MSE jacobian
        dmse = np.tile(2 * dmodel.sigma2.reshape(1, -1).conj().T, (1, nx)) * np.tile(g / dmodel.Ssc[[1], :], (q, 1))

        if q == 1:  # gradient and MSE gradient as a column vector
            or1 = or1.conj().T
            dmse = dmse.conj().T

        # Scaled predictor 
        sy = f @ dmodel.beta + (dmodel.gamma @ r).T

        # Predictor
        y = (dmodel.Ysc[[0], :] + dmodel.Ysc[[1], :] * sy).conj().T

    else:  # several trial sites
        # Get distance to design sites
        dx = np.zeros((mx * m, n))
        kk = np.arange(m).reshape(1, -1)

        for k in np.arange(mx):
            dx[kk, :] = np.tile(x[k, :], (m, 1)) - dmodel.S
            kk = kk + m

        # Get regression function or correlation
        f = regrpoly(x, polynomial=dmodel.regr, jacobian='no')
        r = np.reshape(corr(dmodel.theta, dx, correlation=dmodel.corr, jacobian='no'), (m, mx), order='F')

        # scaled predictor
        sy = f @ dmodel.beta + (dmodel.gamma @ r).T

        # predictor
        y = np.tile(dmodel.Ysc[[0], :], (mx, 1)) + np.tile(dmodel.Ysc[[1], :], (mx, 1)) * sy

        # MSE
        rt = mldivide(dmodel.C, r)
        u = mldivide(dmodel.G, dmodel.Ft.T @ rt - f.T)
        or1 = np.tile(dmodel.sigma2, (mx, 1)) * np.tile((1 + colsum(u ** 2) - colsum(rt ** 2)).conj().T, (1, q))

    return y, or1, or2, dmse


def colsum(x):
    """Columnwise sum of elements in x."""

    if x.shape[0] == 1:
        return x

    else:
        return np.sum(x, axis=0)
