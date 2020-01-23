# from pydace.correlation import corr
# from pydace.regression import regrpoly
# from pydace.utils import lhsdesign
# from pydace.utils.matrixdivide import mldivide, mrdivide
# from pydace.utils.optimizers import BoxMin
import os
import pathlib

import numpy as np
from scipy.linalg import LinAlgError, cholesky, qr
from scipy.spatial.distance import pdist

paths = os.environ['PYTHONPATH'].split(os.pathsep)

pathlib.Path(paths[0]).resolve()

from .correlation import corr
from .regression import regrpoly
from .utils.matrixdivide import mldivide, mrdivide
from .utils.optimizers import BoxMin


class Dace:
    """This the object oriented implementation of the DACE toolbox MATLAB 
    implementation.

    Parameters
    ----------
    regression : str, optional
        Type of mean regression model. Valid models are:

            * Zero order polynomial ('poly0').
            * First order polynomial ('poly1').
            * Second order polynomial ('poly2').

    correlation : str, optional
        Correlation model. Only Gaussian correlation ('corrgauss') implemented.

    optimizer : str, optional
        Type of NLP optimizer used to find the hyperparameters `theta`. Valid 
        optimizers are  'boxmin'. Default is 'boxmin' (original 
        implementation).
    """

    def __init__(self, regression: str, correlation: str,
                 optimizer: str = 'boxmin'):
        self._S = None
        self._Y = None
        self._regression = regression
        self._correlation = correlation
        self._optimizer = optimizer

        self._sampling_method = 'simplicial'
        self._n_points = 100

    # ------------------------------ PROPERTIES -------------------------------
    @property
    def S(self) -> np.ndarray:
        """m-by-n array of design sites (m being the number of samples and n
        the number of input dimensions."""
        return self._S

    @S.setter
    def S(self, value: np.ndarray):
        # normalize the data
        mS = np.mean(value, axis=0)
        sS = np.std(value, axis=0, ddof=1)

        sS[sS == 0] = 1.0

        self._S = (value - mS) / sS

        self.m, self.n = self._S.shape
        self._Ssc = np.vstack((mS, sS))

    @property
    def Y(self) -> np.ndarray:
        """m-by-q array of observed responses (q is the output dimensions).
        q = 1 mean univariate model, otherise the model is considered
        multivariate."""
        return self._Y

    @Y.setter
    def Y(self, value: np.ndarray):
        if value.shape[0] == 1 or value.ndim == 1:
            # if the Y array is row vector or 1D array make it a column vector
            value = value.reshape(-1, 1)

        m, _ = value.shape

        if m != self.S.shape[0]:
            raise ValueError('S and Y must have the same number of rows')

        # normalize the data
        mY = np.mean(value, axis=0)
        sY = np.std(value, axis=0, ddof=1)

        sY[sY == 0] = 1.0

        self._Y = (value - mY) / sY
        self._Ysc = np.vstack((mY, sY))

    # ---------------------------- PRIVATE METHODS ----------------------------
    def _initialize(self) -> dict:
        """Initialize the parameters for the objective function.
        """
        if self.S is None or self.Y is None:
            raise AttributeError("S and/or Y arrays not defined.")

        S = self.S
        Y = self.Y
        sS = np.std(S, axis=0, ddof=1)
        sS[sS == 0] = 1.0

        m, n = S.shape

        # calculate distances D between points
        mzmax = int(m * (m - 1) / 2)  # number of non-zero distances
        D = np.zeros((mzmax, n))  # initialize matrix with distances

        # create indexes for distances between points
        for k in range(n):
            D[:, k] = pdist(S[:, [k]], metric='euclidean')

        if np.min(np.sum(np.abs(D), axis=1), axis=0) == 0:
            raise ValueError('Multiple design sites are not allowed.')

        # regression matrix
        F = regrpoly(S, polynomial=self._regression)  # ignore 'dF'
        mF, p = F.shape

        if mF != m:
            raise ValueError('Number of rows in F and S do not match.')

        if p > mF:
            raise ArithmeticError('Least-squares problem is undetermined.')

        par = {'corr': self._correlation,
               'regr': self._regression,
               'Y': Y,
               'F': F,
               'D': D,
               'sS': sS}

        self._par = par

    def _objfunc(self, theta: np.ndarray):
        """Maximum likelihood evaluation

        Parameters
        ----------
        theta : np.ndarray
            [description]
        """
        # initialize
        obj = np.Inf

        fitpar = {'sigma2': np.NaN,
                  'beta': np.NaN,
                  'gamma': np.NaN,
                  'C': np.NaN,
                  'Ft': np.NaN,
                  'G': np.NaN, }

        m = self._par['F'].shape[0]

        # set up R
        r = corr(theta, self._par['D'], correlation=self._par['corr'])

        mu = (10 + m) * np.spacing(1)
        R = np.triu(np.ones((m, m)), 1)
        R[R == 1.0] = r
        np.fill_diagonal(R, 1.0 + mu)

        try:
            C = cholesky(R).T
        except LinAlgError:
            # cholesky decomp failed, not positive definite, return inf
            self._fitpar = fitpar
            return obj

        # get least squares solution
        Ft = mldivide(C, self._par['F'])
        Q, G = qr(Ft, mode='economic')

        if 1 / np.linalg.cond(G) < 1e-10:
            # check F
            if np.linalg.cond(self._par['F']) > 1e15:
                raise ValueError(
                    ("F is too ill conditioned. Poor combination of "
                     "regression model and design sites."))
            else:  # matrix Ft is too ill conditioned
                self._fitpar = fitpar
                return obj

        Yt = mldivide(C, self._par['Y'])
        beta = mldivide(G, Q.T @ Yt)
        rho = Yt - Ft @ beta
        sigma2 = np.sum(rho ** 2, axis=0) / m
        detR = np.prod(np.diag(C) ** (2 / m), axis=0)
        obj = np.sum(sigma2, axis=0) * detR

        fitpar['sigma2'] = sigma2 * self._Ysc[1, :] ** 2
        fitpar['beta'] = beta
        fitpar['gamma'] = mrdivide(rho.T, C)
        fitpar['C'] = C
        fitpar['Ft'] = Ft
        fitpar['G'] = G.T

        self._fitpar = fitpar
        return obj

    # ---------------------------- PUBLIC METHODS -----------------------------
    def fit(self, S: np.ndarray, Y: np.ndarray, theta0: np.ndarray,
            lob: np.ndarray = None, upb: np.ndarray = None):
        """ Constrained non-linear least-squares fit of a given correlation 
        model to the provided data set and regression model.

        Parameters
        ----------
        S : (M, N) ndarray
            Design sites: an m-by-n array (m being number of samples and n the
            input dimensions).

        Y : (M, Q) ndarray
            Observed responses: m-by-q array (q is the output dimensions).
            q = 1 means univariate, q > 1 means multivariate.

        theta0 : (N,) ndarray
            Correlation function parameters (`theta`). If `lob` and  `upb` are 
            specified, the value of `theta0` is used as initial guess of the 
            optimization problem. Otherwise, the correlation matrix is 
            calculated with the `theta0` given (no optimization).

        lob : (N,) ndarray, optional
            Lower bound of `theta`.

        upb : (N,) ndarray, optional
            Upper bound of `theta`.

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
            When the design sites are too close to one another resulting in 
            ill-conditioning of the correlation matrix.

        """

        self.S = S
        self.Y = Y

        if theta0.ndim != 1:
            theta0 = theta0.flatten()

        if (lob is None and upb is not None) or \
                (lob is not None and upb is None):
            raise ValueError('You must specify both theta bounds.')

        # load objfunc parameters
        self._initialize()

        if lob is not None and upb is not None:
            # optimization
            if lob.ndim != 1 or upb.ndim != 1:
                lob = lob.flatten()
                upb = upb.flatten()

            lth = theta0.size

            if lob.size != lth or upb.size != lth:
                raise ValueError(
                    'theta0, lob and upb must have the same length.')

            any_lob_le_zero = np.any(np.less_equal(lob, np.zeros(lob.shape)))
            any_upb_lt_lob = np.any(np.less(upb, lob))
            if any_lob_le_zero or any_upb_lt_lob:
                raise ValueError('The bounds must satisfy 0 < lob <= upb.')

            if self._optimizer == 'boxmin':
                # optimize using boxmin
                bmin = BoxMin(self, theta0, lob, upb)

                if np.isinf(bmin.f):
                    raise ValueError(
                        'Bad parameter region. Try increasing upb.')

                f = bmin.f
                theta = bmin.t
                self._objfunc(theta)
                perf = bmin.perf_info
                self.theta = theta

            else:
                raise NotImplementedError(
                    "Invalid optimizer option or not implemented.")
        else:  # given theta
            if np.any(np.less_equal(theta0, np.zeros(theta0.shape))):
                raise ValueError('theta0 must be strictly positive.')

            f = self._objfunc(theta0)
            self.theta = theta0

    def predict(self, X: np.ndarray, compute_jacobian=False, compute_mse=False,
                compute_mse_jacobian=False):
        """
        Predictor for y(x) using the given DACE model.

        Parameters
        ----------
        x : (M, N) ndarray
            Trial design sites with n dimensions.

        compute_jacobian : bool, optional
            Whether or not to compute the jacobian.

        compute_mse : bool, optional
            Whether or not to compute the Mean Squared Error (MSE) of the 
            prediction

        compute_mse_jacobian : bool, optional
            Whether or not to compute the MSE jacobian of the prediction.

        Returns
        -------
        y : (M,1) ndarray
            Predicted response at x.

        or1 : (M, ...) ndarray
            If m = 1, and `compute_jacobian` is set to True, then `or1` is a 
            gradient vector/Jacobian matrix of predictor. Otherwise, `or1` is a 
            vector with m rows containing the estimated mean squared error of 
            the predictor.

        or2 : float
            If m = 1, and `compute_mse` is set to True, then `or2` is a the 
            estimated mean squared error (MSE) of the predictor. Otherwise 
            (m > 1 or `compute_mse' set to False), it's a NaN type.

        dmse : (M, ...) ndarray
            The gradient vector/Jacobian Matrix of the MSE. Only available when
            m = 1 and `compute_mse_jacobian` set to True, otherwise it's a NaN 
            type.

        """
        x = X  # to avoid confusion with internal X (train X)

        if x.ndim == 1:  # change from 1d array to 2d
            x = x[np.newaxis, :]

        if x.ndim > 2:
            raise ValueError(
                "Input arrays of dimension higher than 2 aren't allowed.")

        if np.all(np.isnan(self._fitpar['beta'])):
            raise ValueError(
                "Kriging build is invalid because it contains NaN values.")

        m, n = self.m, self.n
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
        x = (x - self._Ssc[[0], :]) / self._Ssc[[1], :]
        q = self._Ysc.shape[1]  # number of response functions

        if mx == 1:  # one site only
            dx = x - self.S  # distance to design sites

            # get correlation a regression data depending whether or not
            # jacobian info was required
            if compute_jacobian or compute_mse_jacobian:  # jacobian required
                f, df = regrpoly(x, polynomial=self._regression, jacobian=True)
                r, dr = corr(self.theta, dx, correlation=self._correlation,
                             jacobian=True)
            else:
                f = regrpoly(x, polynomial=self._regression)
                r = corr(self.theta, dx, correlation=self._correlation)

            # compute the prediction
            # Scaled predictor
            sy = f @ self._fitpar['beta'] + (self._fitpar['gamma'] @ r).T

            # Predictor
            if q == 1:  # make sure the return is a scalar
                y = np.asscalar((self._Ysc[[0], :] + self._Ysc[[1], :] * sy).T)
            else:  # otherwise, keep it as it is
                y = (self._Ysc[[0], :] + self._Ysc[[1], :] * sy).T

            # compute the prediction jacobian
            if compute_jacobian:
                # scaled jacobian
                sdy = np.transpose(df @ self._fitpar['beta']) + \
                    self._fitpar['gamma'] @ dr

                # unscaled jacobian
                dy = sdy * self._Ysc[[1], :].T / self._Ssc[[1], :]

                if q == 1:  # gradient as column vector for single dimension
                    dy = dy.T

            # compute MSE
            if compute_mse:
                # MSE
                rt = mldivide(self._fitpar['C'], r)
                u = self._fitpar['Ft'].T @ rt - f.T
                v = mldivide(self._fitpar['G'], u)
                mse = np.tile(self._fitpar['sigma2'], (mx, 1)) * \
                    np.tile((1 + np.sum(v ** 2, axis=0) -
                             np.sum(rt ** 2, axis=0)).T,
                            (1, q))

                if q == 1:  # make sure the return is a scalar if q == 1
                    mse = np.asscalar(mse)

                # compute MSE jacobian
                if compute_mse:
                    # scaled gradient as row vector
                    Gv = mldivide(self._fitpar['G'].T, v)
                    g = (self._fitpar['Ft'] @ Gv - rt).T @ \
                        mldivide(self._fitpar['C'], dr) - (df @ Gv).T

                    # unscaled MSE jacobian
                    dmse = np.tile(2 * self._fitpar['sigma2'].reshape(1, -1).T,
                                   (1, nx)) * \
                        np.tile(g / self._Ssc[[1], :], (q, 1))

                    if q == 1:  # gradient as column vector for single dimension
                        dmse = dmse.conj().T

        else:  # several trial sites

            if compute_jacobian or compute_mse_jacobian:  # basic sanitation
                raise ValueError(("Can't compute either prediction or MSE "
                                  "jacobian for several design sites."))

            # Get distance to design sites
            dx = np.zeros((mx * m, n))
            kk = np.arange(m).reshape(1, -1)

            for k in np.arange(mx):
                dx[kk, :] = x[k, :] - self.S
                kk = kk + m

            # Get regression function or correlation
            f = regrpoly(x, polynomial=self._regression)
            r = np.reshape(corr(self.theta, dx, correlation=self._correlation),
                           (m, mx), order='F')

            # scaled predictor
            sy = f @ self._fitpar['beta'] + (self._fitpar['gamma'] @ r).T

            # predictor
            y = self._Ysc[[0], :] + self._Ysc[[1], :] * sy

            # MSE
            if compute_mse:
                rt = mldivide(self._fitpar['C'], r)
                u = mldivide(self._fitpar['G'],
                             self._fitpar['Ft'].T @ rt - f.T)

                mse = self._fitpar['sigma2'] * \
                    (1 + self._colsum(u ** 2) - self._colsum(rt ** 2)).T

        return y, dy, mse, dmse

    def _colsum(self, x):
        """Columnwise sum of elements in x."""

        if x.shape[0] == 1:
            return x

        else:
            return np.sum(x, axis=0)
