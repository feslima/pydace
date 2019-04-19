import numpy as np

from pydace.aux_functions.matrixdivide import mldivide
from pydace.corr import corr
from pydace.regr import regrpoly
from pydace.boxmin import objfunc, boxmin


def dacefit(S, Y, regr, corr, theta0, lob=None, upb=None):
    """ Constrained non-linear least-squares fit of a given correlation model to the provided data set and regression
    model.

    Parameters
    ----------
    S : (M, N) ndarray
        Design sites: an m-by-n array (m being number of samples and n the input dimensions).
    Y : (M, Q) ndarray
        Observed responses: m-by-q array (q is the output dimensions).
        q = 1 means univariate, q > 1 means multivariate
    regr : {'poly0', 'poly1', 'poly2'}
        Regression model.
        Valid model are:
            Zero order polynomial ('poly0').
            First order polynomial ('poly1').
            Second order polynomial ('poly2').
    corr : {'corrgauss'}
        Correlation model. Only Gaussian correlation ('corrgauss') implemented.
    theta0 : (1, N) or (N,1) ndarray
        Correlation function parameters (:math:`\\theta`). If `lob` and  `upb` are specified, the value of `theta0` is
        used as initial guess of the optimization problem. Otherwise, the correlation matrix is calculated with the
        `theta0` given (no optimization).
    lob : (1, N) or (N,1) ndarray, optional
        Lower bound of :math:`\\theta`.
    upb : (1, N) or (N,1) ndarray, optional
        Upper bound of :math:`\\theta`.

    Returns
    -------
    dacemodel : dict
        DaceModel object containing all the info necessary to be used by the predictor function.
    perf : dict
        PerformanceInfo dictionary containing information about the optimzation.

    Raises
    ------
    ValueError
        When `S` and `Y` does not have the same number of rows.
        When specifying only one of either `lob` and `upb`.
        When `lob`, `upb` and `theta0` does not have the same lengths.
        When `theta0` is outside the bound `lob`<`theta`<=`upb`.
        When `theta0` <= 0.
        When `S` has any repeated rows.
    ArithmeticError
        When the optimization fails.
    LinAlgError
        When the design sites are too close to one another resulting in ill-conditioning of the correlation matrix.
    """
    if Y.ndim == 1:  # if observed data is a column vector, make sure it is a matrix (ndim = 2)
        Y = Y.reshape(-1, 1)

    # check design points
    m, n = S.shape  # number of design sites and their dimension
    sY = Y.shape

    if np.min(sY, axis=0) == 1:
        Y = Y.reshape(-1, 1)
        lY = np.max(sY, axis=0)
        sY = Y.shape
    else:
        lY = sY[0]

    if m != lY:
        raise ValueError('S and Y must have the same number of rows.')

    # check correlation parameters
    lth = theta0.size

    if (lob is None and upb is not None) or (lob is not None and upb is None):
        raise ValueError('You must specify both theta bounds.')

    if lob is not None and upb is not None:  # optimization case

        if lob.size != lth or upb.size != lth:
            raise ValueError('theta0, lob and upb must have the same length.')

        any_lob_le_zero = np.any(np.less_equal(lob, np.zeros(lob.shape)))
        any_upb_lt_lob = np.any(np.less(upb, lob))
        if any_lob_le_zero or any_upb_lt_lob:
            raise ValueError('The bounds must satisfy 0 < lob <= upb.')

    else:  # given theta
        if np.any(np.less_equal(theta0, np.zeros(theta0.shape))):
            raise ValueError('theta0 must be strictly positive.')

    # normalize the data
    mS = np.mean(S, axis=0)
    sS = np.std(S, axis=0, ddof=1)  # setting the delta degrees of fredom to 1 (matlab default)

    mY = np.mean(Y, axis=0)
    sY = np.std(Y, axis=0, ddof=1)

    sS[sS == 0] = 1
    sY[sY == 0] = 1

    S = (S - mS) / sS
    Y = (Y - mY) / sY

    # calculate distances D between points
    mzmax = int(m * (m - 1) / 2)  # number of non-zero distances
    ij = np.zeros((mzmax, 2), dtype=np.int)  # initialize matrix with indices
    D = np.zeros((mzmax, n))  # initialize matrix with distances
    ll = np.zeros(m - 1, dtype=np.int)

    for k in np.arange(1, m):
        ll = ll[-1] + np.arange(1, m - k + 1, dtype=np.int)
        ij[ll - 1, :] = np.hstack((np.tile(k - 1, (m - k, 1)), np.arange(k + 1, m + 1).reshape(-1, 1) - 1))
        D[ll - 1, :] = np.tile(S[k - 1, :], (m - k, 1)) - S[np.arange(k, m), :]

    if np.min(np.sum(np.abs(D), axis=1), axis=0) == 0:
        raise ValueError('Multiple design sites are not allowed.')

    # regression matrix
    F = regrpoly(S, polynomial=regr)  # ignore 'dF'
    mF, p = F.shape

    if mF != m:
        raise ValueError('Number of rows in F and S do not match.')

    if p > mF:
        raise ArithmeticError('Least-squares problem is undetermined.')

    # parameters for objective function
    # par = ObjPar(corr, regr, Y, F, D, ij, sS)
    par = {'corr': corr,
           'regr': regr,
           'Y': Y,
           'F': F,
           'D': D,
           'ij': ij,
           'sS': sS}

    if lob is not None and upb is not None:
        theta, f, fit, perf = boxmin(theta0, lob, upb, par)

        if np.isinf(f):
            raise ValueError('Bad parameter region. Try increasing upb.')

    elif lob is None and upb is None:
        # Given theta
        theta = theta0.reshape(-1, 1)
        f, fit = objfunc(theta, par)
        perf = {'nv': 1,
                'perf': np.vstack((theta, f, 1))}

        if np.isinf(f):
            raise ValueError('Poor theta value. ')

    # return values
    # return DaceModel(regr, corr, theta, sY, fit, S,
    #                  np.vstack((mS, sS)), np.vstack((mY, sY))), perf
    return {'regr': regr,
            'corr': corr,
            'theta': theta.T,
            'beta': fit['beta'],
            'gamma': fit['gamma'],
            'sigma2': sY ** 2 * fit['sigma2'],
            'S': S,
            'Ssc': np.vstack((mS, sS)),
            'Ysc': np.vstack((mY, sY)),
            'C': fit['C'],
            'Ft': fit['Ft'],
            'G': fit['G']}, perf


def predictor(x, dmodel, compute_jacobian=False, compute_mse=False, compute_mse_jacobian=False):
    """
    Predictor for y(x) using the given DACE model. (Optimized version to enable fast calculations)

    Parameters
    ----------
    x : (M, N) ndarray
        Trial design sites with n dimensions.
    dmodel : dict
        DACE model struct.
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

    if np.all(np.isnan(dmodel['beta'])):
        raise ValueError("Kriging build is invalid because it contains NaN values.")

    m, n = dmodel['S'].shape  # number of design sites and number of dimension
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
    x = (x - dmodel['Ssc'][[0], :]) / dmodel['Ssc'][[1], :]
    q = dmodel['Ysc'].shape[1]  # number of response functions

    if mx == 1:  # one site only
        dx = x - dmodel['S']  # distance to design sites

        # get correlation a regression data depending whether or not jacobian info was required
        if compute_jacobian or compute_mse_jacobian:  # jacobian required
            f, df = regrpoly(x, polynomial=dmodel['regr'], jacobian=True)
            r, dr = corr(dmodel['theta'], dx, correlation=dmodel['corr'], jacobian=True)

        else:
            f = regrpoly(x, polynomial=dmodel['regr'])
            r = corr(dmodel['theta'], dx, correlation=dmodel['corr'])

        # compute the prediction
        # Scaled predictor
        sy = f @ dmodel['beta'] + (dmodel['gamma'] @ r).T

        # Predictor
        if q == 1:  # make sure the return is a scalar
            y = np.asscalar((dmodel['Ysc'][[0], :] + dmodel['Ysc'][[1], :] * sy).conj().T)
        else:  # otherwise, keep it as it is
            y = (dmodel['Ysc'][[0], :] + dmodel['Ysc'][[1], :] * sy).conj().T

        # compute the prediction jacobian
        if compute_jacobian:
            # scaled jacobian
            sdy = np.transpose(df @ dmodel['beta']) + dmodel['gamma'] @ dr

            # unscaled jacobian
            dy = sdy * dmodel['Ysc'][[1], :].T / dmodel['Ssc'][[1], :]

            if q == 1:  # gradient as column vector for single dimension
                dy = dy.conj().T

        # compute MSE
        if compute_mse:
            # MSE
            rt = mldivide(dmodel['C'], r)
            u = dmodel['Ft'].T @ rt - f.T
            v = mldivide(dmodel['G'], u)
            mse = np.tile(dmodel['sigma2'], (mx, 1)) * np.tile(
                (1 + np.sum(v ** 2, axis=0) - np.sum(rt ** 2, axis=0)).conj().T,
                (1, q))

            if q == 1:  # make sure the return is a scalar if q == 1
                mse = np.asscalar(mse)

            # compute MSE jacobian
            if compute_mse:
                # scaled gradient as row vector
                Gv = mldivide(dmodel['G'].conj().T, v)
                g = (dmodel['Ft'] @ Gv - rt).conj().T @ mldivide(dmodel['C'], dr) - (df @ Gv).conj().T

                # unscaled MSE jacobian
                dmse = np.tile(2 * dmodel['sigma2'].reshape(1, -1).conj().T, (1, nx)) * np.tile(
                    g / dmodel['Ssc'][[1], :], (q, 1))

                if q == 1:  # gradient as column vector for single dimension
                    dmse = dmse.conj().T

    else:  # several trial sites

        if compute_jacobian or compute_mse_jacobian:  # basic sanitation
            raise ValueError("Can't compute either prediction or MSE jacobian for several design sites.")

        # Get distance to design sites
        dx = np.zeros((mx * m, n))
        kk = np.arange(m).reshape(1, -1)

        for k in np.arange(mx):
            dx[kk, :] = x[k, :] - dmodel['S']
            kk = kk + m

        # Get regression function or correlation
        f = regrpoly(x, polynomial=dmodel['regr'])
        r = np.reshape(corr(dmodel['theta'], dx, correlation=dmodel['corr']), (m, mx), order='F')

        # scaled predictor
        sy = f @ dmodel['beta'] + (dmodel['gamma'] @ r).T

        # predictor
        # org -- y = np.tile(dmodel['Ysc'][[0], :], (mx, 1)) + np.tile(dmodel['Ysc'][[1], :], (mx, 1)) * sy
        y = dmodel['Ysc'][[0], :] + dmodel['Ysc'][[1], :] * sy

        # MSE
        rt = mldivide(dmodel['C'], r)
        u = mldivide(dmodel['G'], dmodel['Ft'].T @ rt - f.T)
        # org --
        # mse = np.tile(dmodel['sigma2'], (mx, 1)) * np.tile((1 + colsum(u ** 2) - colsum(rt ** 2)).conj().T,(1, q))
        mse = dmodel['sigma2'] * (1 + _colsum(u ** 2) - _colsum(rt ** 2)).conj().T

    return y, dy, mse, dmse


def _colsum(x):
    """Columnwise sum of elements in x."""

    if x.shape[0] == 1:
        return x

    else:
        return np.sum(x, axis=0)
