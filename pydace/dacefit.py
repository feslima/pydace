import numpy as np
import scipy as sp

from aux_functions.matrixdivide import mldivide, mrdivide


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
    dacemodel : DaceModel
        DaceModel object containing all the info necessary to be used by the predictor function.
    perf : PerformanceInfo
        PerformanceInfo object containing information about the optimzation.

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
    par = ObjPar(corr, regr, Y, F, D, ij, sS)

    if lob is not None and upb is not None:
        theta, f, fit, perf = boxmin(theta0, lob, upb, par)

        if np.isinf(f):
            raise ValueError('Bad parameter region. Try increasing upb.')

    elif lob is None and upb is None:
        # Given theta
        theta = theta0.reshape(-1, 1)
        f, fit = objfunc(theta, par)
        perf = PerformanceInfo(1, np.vstack((theta, f, 1)))

        if np.isinf(f):
            raise ValueError('Poor theta value. ')

    # return values
    return DaceModel(regr, corr, theta, sY, fit, S,
                     np.vstack((mS, sS)), np.vstack((mY, sY))), perf


def objfunc(theta, par):
    # initialize
    obj = np.Inf

    fit = FitObj(np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN)
    m = par.F.shape[0]

    # set up R
    r = corr(theta, par.D, correlation=par.corr)
    idx = np.nonzero(r > 0)
    o = np.arange(m).conj().T
    mu = (10 + m) * np.spacing(1)
    R = np.zeros((m, m))
    R[np.vstack((par.ij[idx[0], 0].reshape(-1, 1), o.reshape(-1, 1))), np.vstack(
        (par.ij[idx[0], 1].reshape(-1, 1), o.reshape(-1, 1)))] = np.vstack(
        (r[idx[0]].reshape(-1, 1), np.ones((m, 1)) + mu))
    try:
        # using scipy's cholesky because numpy's does not produce expected values
        C = sp.linalg.cholesky(R).T
    except sp.linalg.LinAlgError:
        return obj, fit  # not positive definite, return inf value

    # get least squares solution
    Ft = mldivide(C, par.F)
    Q, G = sp.linalg.qr(Ft, mode='economic')

    if 1 / np.linalg.cond(G) < 1e-10:
        # check F
        if np.linalg.cond(par.F) > 1e15:
            raise ValueError('F is too ill conditioned. Poor combination of regression model and design sites.')
        else:  # matrix Ft is too ill conditioned
            return obj, fit

    Yt = mldivide(C, par.Y)
    beta = mldivide(G, Q.T @ Yt)
    rho = Yt - Ft @ beta
    sigma2 = np.sum(rho ** 2, axis=0) / m
    detR = np.prod(np.diag(C) ** (2 / m), axis=0)
    obj = np.sum(sigma2, axis=0) * detR

    fit.sigma2 = sigma2
    fit.beta = beta
    fit.gamma = mrdivide(rho.T, C)
    fit.C = C
    fit.Ft = Ft
    fit.G = G.T

    return obj, fit


def boxmin(t0, lo, up, par):
    # initialize
    t, f, fit, itpar = start(t0, lo, up, par)

    if not np.isinf(f):
        # Iterate
        p = t.size
        if p <= 2:
            kmax = 2
        else:
            kmax = np.minimum(p, 4)

        for k in np.arange(kmax):
            th = t.copy()
            t, f, fit, itpar = explore(t, f, fit, itpar, par)
            t, f, fit, itpar = move(th, t, f, fit, itpar, par)

    perf = PerformanceInfo(itpar.nv, itpar.perf[:, 0:itpar.nv])

    return t, f, fit, perf


def start(t0, lo, up, par):
    # get starting point and iteration parameters

    # Initialize
    t = t0.reshape(-1, 1)
    lo = lo.reshape(-1, 1)
    up = up.reshape(-1, 1)
    p = t.size

    D = 2 ** (np.arange(1, p + 1).reshape(-1, 1) / (p + 2))

    ee = np.nonzero(np.equal(up, lo))  # Equality constraints
    if ee[0].size != 0:
        D[ee] = np.ones((ee[0].size, 1))
        t[ee] = up[ee]

    # Free starting values
    ng = np.logical_or(np.less(t, lo), np.less(up, t))
    ng = (np.nonzero(ng))

    t[ng] = (lo[ng] * up[ng] ** 7) ** (1 / 8)  # Starting point

    ne = np.nonzero(D != 1)

    # Check starting point and initialize perfomance info
    f, fit = objfunc(t, par)
    nv = 1
    itpar = IterationParameters(D, ne, lo, up, np.zeros((p + 2, 200 * p)), 1)
    itpar.perf[:, [0]] = np.vstack((t, f, 1))

    if np.isinf(f):  # Bad parameter region
        return t, f, fit, itpar

    if ng[0].size > 1:  # Try to improve starting guess

        d0 = 16
        d1 = 2
        q = ng[0].size
        th = t.copy()
        fh = f
        jdom = ng[0][0]

        for k in np.arange(q):
            j = ng[0][k]
            fk = fh
            tk = th

            DD = np.ones((p, 1))
            DD[ng[0]] = np.tile(1 / d1, (q, 1))
            DD[j] = 1 / d0

            alpha = np.min(np.log(lo[ng[0]] / th[ng[0]]) / np.log(DD[ng[0]])) / 5
            v = DD ** alpha
            tk = th

            for rept in np.arange(4):
                tt = tk * v
                ff, fitt = objfunc(tt, par)
                nv += 1
                itpar.perf[:, [nv - 1]] = np.vstack((tt, ff, 1))

                if ff <= fk:
                    tk = tt.copy()
                    fk = ff

                    if ff <= f:
                        t = tt.copy()
                        f = ff
                        fit = fitt
                        jdom = j.copy()

                else:
                    itpar.perf[[-1], [nv - 1]] = -1
                    break

        # Update data
        if jdom > 0:
            D[np.hstack((0, jdom - 1))] = D[np.hstack((jdom - 1, 0))]
            itpar.D = D

    itpar.nv = nv
    return t, f, fit, itpar


def explore(t, f, fit, itpar, par):
    # explore step

    nv = itpar.nv
    ne = itpar.ne

    for k in np.arange(ne[0].size):
        j = ne[0][k]
        tt = t.copy()
        DD = itpar.D[j]

        if t[j] == itpar.up[j]:
            atbd = True
            tt[j] = t[j] / np.sqrt(DD)

        elif t[j] == itpar.lo[j]:
            atbd = True
            tt[j] = t[j] * np.sqrt(DD)

        else:
            atbd = False
            tt[j] = np.minimum(itpar.up[j], t[j] * DD)

        ff, fitt = objfunc(tt, par)
        nv += 1
        itpar.perf[:, [nv - 1]] = np.vstack((tt, ff, 2))

        if ff < f:
            t = tt.copy()
            f = ff
            fit = fitt

        else:
            itpar.perf[-1, [nv - 1]] = -2

            if not atbd:  # try decrease

                tt[j] = np.maximum(itpar.lo[j], t[j] / DD)
                ff, fitt = objfunc(tt, par)
                nv += 1
                itpar.perf[:, [nv - 1]] = np.vstack((tt, ff, 2))

                if ff < f:
                    t = tt.copy()
                    f = ff
                    fit = fitt

                else:
                    itpar.perf[-1, [nv - 1]] = -2

    itpar.nv = nv
    return t, f, fit, itpar


def move(th, t, f, fit, itpar, par):
    # Pattern move
    nv = itpar.nv
    p = t.size

    v = t / th

    if np.all(v == 1):
        itpar.D = itpar.D[np.r_[1:p, 0]] ** 0.2
        return t, f, fit, itpar

    # proper move
    rept = True
    while rept:
        tt = np.minimum(itpar.up, np.maximum(itpar.lo, t * v))
        ff, fitt = objfunc(tt, par)
        nv += 1
        itpar.perf[:, [nv - 1]] = np.vstack((tt, ff, 3))

        if ff < f:
            t = tt.copy()
            f = ff
            fit = fitt
            v = v ** 2

        else:
            itpar.perf[-1, [nv - 1]] = -3
            rept = False

        if np.any(np.logical_or(np.equal(tt, itpar.lo), np.equal(tt, itpar.up))):
            rept = False

    itpar.nv = nv
    itpar.D = itpar.D[np.r_[1:p, 0]] ** 0.25
    return t, f, fit, itpar


# @profile  # line_profiler
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


# @profile  # line_profiler
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
        raise ValueError("Invalid jacobian option. Valid values are True and False.")

    m, n = d.shape  # number of differences and dimension of data

    if correlation == 'corrgauss':
        if theta.size == 1:
            theta = np.tile(theta, (1, n))
        elif theta.size != n:
            raise ValueError(f'Length of theta must be 1 or {n}.')

        # original -- td = d ** 2 * np.tile(-theta.reshape(-1, 1).T, (m, 1))
        td = d * -theta.flatten()
        r = np.exp(np.sum(d * td, axis=1, keepdims=True))  # keepdims=True is to force r to be a matrix column

        if jacobian:
            # original -- dr = np.tile(-2 * theta.reshape(-1, 1).T, (m, 1)) * d * np.tile(r, (1, n))
            dr = 2 * td * r

    else:
        raise NotImplementedError('Correlation not implemented or invalid.')

    if jacobian:
        return r, dr
    else:
        return r


class ObjPar(object):

    def __init__(self, corr, regr, Y, F, D, ij, sS):
        self.corr = corr
        self.regr = regr
        self.Y = Y
        self.F = F
        self.D = D
        self.ij = ij
        self.sS = sS


class IterationParameters(object):

    def __init__(self, D, ne, lo, up, perf, nv):
        self.D = D
        self.ne = ne
        self.lo = lo
        self.up = up
        self.perf = perf
        self.nv = nv


class FitObj(object):

    def __init__(self, sigma2, beta, gamma, C, Ft, G):
        self.sigma2 = sigma2
        self.beta = beta
        self.gamma = gamma
        self.C = C
        self.Ft = Ft
        self.G = G


class PerformanceInfo(object):
    # TODO: implement docstring

    def __init__(self, nv, perf):
        self.nv = nv
        self.perf = perf


class DaceModel(object):
    """
    DACE model object. Generated by dacefit.

    Attributes
    ----------
    regr : str
        Type of regression model function. Valid values are: 'regpoly0', 'regpoly1', 'regpoly2'.
    corr : str
        Type of correlation model function. Valid values are: 'corrgauss'.
    theta : ndarray
        Correlation function hyperparameters.
    beta : float
        Generalized least-squares estimate.
    gamma : ndarray
        Correlation factors.
    sigma2 : float
        Estimate of the process variance.
    S : ndarray
        Scaled design sites.
    Ssc : ndarray
        2-by-n array with scaling factors for design sites.
    Ysc : ndarray
        2-by-q array with scaling factors for design responses.
    C : ndarray
        Cholesky factor of correlation matrix.
    Ft : ndarray
        Decorrelated regression matrix.
    G : ndarray
        Matrix G. See dace documentation for details.
    """
    def __init__(self, regr, corr, theta, sY, fit, S, Ssc, Ysc):
        self.regr = regr
        self.corr = corr
        self.theta = theta.T
        self.beta = fit.beta
        self.gamma = fit.gamma
        self.sigma2 = sY ** 2 * fit.sigma2
        self.S = S
        self.Ssc = Ssc
        self.Ysc = Ysc
        self.C = fit.C
        self.Ft = fit.Ft
        self.G = fit.G
