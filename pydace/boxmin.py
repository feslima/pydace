import numpy as np
import scipy as sp

from aux_functions.matrixdivide import mldivide, mrdivide
from pydace import corr


def objfunc(theta, par):
    # initialize
    obj = np.Inf

    # fit = FitObj(np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN)
    fit = {'sigma2': np.NaN,
           'beta': np.NaN,
           'gamma': np.NaN,
           'C': np.NaN,
           'Ft': np.NaN,
           'G': np.NaN, }
    m = par['F'].shape[0]

    # set up R
    r = corr(theta, par['D'], correlation=par['corr'])
    idx = np.nonzero(r > 0)
    o = np.arange(m).conj().T
    mu = (10 + m) * np.spacing(1)
    R = np.zeros((m, m))
    R[np.vstack((par['ij'][idx[0], 0].reshape(-1, 1), o.reshape(-1, 1))), np.vstack(
        (par['ij'][idx[0], 1].reshape(-1, 1), o.reshape(-1, 1)))] = np.vstack(
        (r[idx[0]].reshape(-1, 1), np.ones((m, 1)) + mu))
    try:
        # using scipy's cholesky because numpy's does not produce expected values
        C = sp.linalg.cholesky(R).T
    except sp.linalg.LinAlgError:
        return obj, fit  # not positive definite, return inf value

    # get least squares solution
    Ft = mldivide(C, par['F'])
    Q, G = sp.linalg.qr(Ft, mode='economic')

    if 1 / np.linalg.cond(G) < 1e-10:
        # check F
        if np.linalg.cond(par['F']) > 1e15:
            raise ValueError('F is too ill conditioned. Poor combination of regression model and design sites.')
        else:  # matrix Ft is too ill conditioned
            return obj, fit

    Yt = mldivide(C, par['Y'])
    beta = mldivide(G, Q.T @ Yt)
    rho = Yt - Ft @ beta
    sigma2 = np.sum(rho ** 2, axis=0) / m
    detR = np.prod(np.diag(C) ** (2 / m), axis=0)
    obj = np.sum(sigma2, axis=0) * detR

    fit['sigma2'] = sigma2
    fit['beta'] = beta
    fit['gamma'] = mrdivide(rho.T, C)
    fit['C'] = C
    fit['Ft'] = Ft
    fit['G'] = G.T

    return obj, fit


def boxmin(t0, lo, up, par):
    # initialize
    t, f, fit, itpar = _start(t0, lo, up, par)

    if not np.isinf(f):
        # Iterate
        p = t.size
        if p <= 2:
            kmax = 2
        else:
            kmax = np.minimum(p, 4)

        for k in np.arange(kmax):
            th = t.copy()
            t, f, fit, itpar = _explore(t, f, fit, itpar, par)
            t, f, fit, itpar = move(th, t, f, fit, itpar, par)

    # perf = PerformanceInfo(itpar['nv'], itpar['perf'][:, 0:itpar['nv']])
    perf = {'nv': itpar['nv'],
            'perf': itpar['perf'][:, 0:itpar['nv']]}

    return t, f, fit, perf


def _start(t0, lo, up, par):
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
    # itpar = IterationParameters(D, ne, lo, up, np.zeros((p + 2, 200 * p)), 1)
    itpar = {'D': D,
             'ne': ne,
             'lo': lo,
             'up': up,
             'perf': np.zeros((p + 2, 200 * p)),
             'nv': 1}
    itpar['perf'][:, [0]] = np.vstack((t, f, 1))

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
                itpar['perf'][:, [nv - 1]] = np.vstack((tt, ff, 1))

                if ff <= fk:
                    tk = tt.copy()
                    fk = ff

                    if ff <= f:
                        t = tt.copy()
                        f = ff
                        fit = fitt
                        jdom = j.copy()

                else:
                    itpar['perf'][[-1], [nv - 1]] = -1
                    break

        # Update data
        if jdom > 0:
            D[np.hstack((0, jdom - 1))] = D[np.hstack((jdom - 1, 0))]
            itpar['D'] = D

    itpar['nv'] = nv
    return t, f, fit, itpar


def _explore(t, f, fit, itpar, par):
    # explore step

    nv = itpar['nv']
    ne = itpar['ne']

    for k in np.arange(ne[0].size):
        j = ne[0][k]
        tt = t.copy()
        DD = itpar['D'][j]

        if t[j] == itpar['up'][j]:
            atbd = True
            tt[j] = t[j] / np.sqrt(DD)

        elif t[j] == itpar['lo'][j]:
            atbd = True
            tt[j] = t[j] * np.sqrt(DD)

        else:
            atbd = False
            tt[j] = np.minimum(itpar['up'][j], t[j] * DD)

        ff, fitt = objfunc(tt, par)
        nv += 1
        itpar['perf'][:, [nv - 1]] = np.vstack((tt, ff, 2))

        if ff < f:
            t = tt.copy()
            f = ff
            fit = fitt

        else:
            itpar['perf'][-1, [nv - 1]] = -2

            if not atbd:  # try decrease

                tt[j] = np.maximum(itpar['lo'][j], t[j] / DD)
                ff, fitt = objfunc(tt, par)
                nv += 1
                itpar['perf'][:, [nv - 1]] = np.vstack((tt, ff, 2))

                if ff < f:
                    t = tt.copy()
                    f = ff
                    fit = fitt

                else:
                    itpar['perf'][-1, [nv - 1]] = -2

    itpar['nv'] = nv
    return t, f, fit, itpar


def move(th, t, f, fit, itpar, par):
    # Pattern move
    nv = itpar['nv']
    p = t.size

    v = t / th

    if np.all(v == 1):
        itpar['D'] = itpar['D'][np.r_[1:p, 0]] ** 0.2
        return t, f, fit, itpar

    # proper move
    rept = True
    while rept:
        tt = np.minimum(itpar['up'], np.maximum(itpar['lo'], t * v))
        ff, fitt = objfunc(tt, par)
        nv += 1
        itpar['perf'][:, [nv - 1]] = np.vstack((tt, ff, 3))

        if ff < f:
            t = tt.copy()
            f = ff
            fit = fitt
            v = v ** 2

        else:
            itpar['perf'][-1, [nv - 1]] = -3
            rept = False

        if np.any(np.logical_or(np.equal(tt, itpar['lo']), np.equal(tt, itpar['up']))):
            rept = False

    itpar['nv'] = nv
    itpar['D'] = itpar['D'][np.r_[1:p, 0]] ** 0.25
    return t, f, fit, itpar
