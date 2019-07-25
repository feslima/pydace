import numpy as np


class BoxMin:
    """BoxMin optimizer used to find the hyperparameters of the Dace class 
    (fit).
    """

    def __init__(self, dace_object, theta0: np.ndarray,
                 theta_lower: np.ndarray, theta_upper: np.ndarray):
        # initialize
        self.dace = dace_object
        self._start(theta0, theta_lower, theta_upper)

        if not np.isinf(self.f):
            # Iterate
            p = self.t.size
            if p <= 2:
                kmax = 2
            else:
                kmax = np.minimum(p, 4)

            for k in np.arange(kmax):
                th = self.t.copy()
                self._explore()
                self._move(th)

        self.perf_info = {'nv': self.nv,
                          'perf': self.perf[:, 0:self.nv]}

    def _start(self, t0: np.ndarray, lo: np.ndarray, up: np.ndarray):
        # get starting point and iteration parameters
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
        f = self.dace._objfunc(t)
        nv = 1

        # itpar
        self.D = D
        self.ne = ne
        self.lo = lo
        self.up = up
        self.perf = np.zeros((p + 2, 200 * p))
        self.nv = nv
        self.perf[:, [0]] = np.vstack((t, f, 1))

        # other params
        self.f = f
        self.t = t

        if np.isinf(f):  # bad parameter region
            return

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

                alpha = np.min(
                    np.log(lo[ng[0]] / th[ng[0]]) / np.log(DD[ng[0]])) / 5
                v = DD ** alpha
                tk = th

                for rept in np.arange(4):
                    tt = tk * v
                    ff = self.dace._objfunc(tt)
                    nv += 1
                    self.perf[:, [nv - 1]] = np.vstack((tt, ff, 1))

                    if ff <= fk:
                        tk = tt.copy()
                        fk = ff

                        if ff <= f:
                            t = tt.copy()
                            f = ff
                            jdom = j.copy()

                    else:
                        self.perf[[-1], [nv - 1]] = -1
                        break

            # Update data
            if jdom > 0:
                D[np.hstack((0, jdom - 1))] = D[np.hstack((jdom - 1, 0))]
                self.D = D

        self.nv = nv

        # other params
        self.f = f
        self.t = t

    def _explore(self):
        # explore step

        nv = self.nv
        ne = self.ne

        t = self.t
        f = self.f

        for k in np.arange(ne[0].size):
            j = ne[0][k]
            tt = t.copy()
            DD = self.D[j]

            if t[j] == self.up[j]:
                atbd = True
                tt[j] = t[j] / np.sqrt(DD)

            elif t[j] == self.lo[j]:
                atbd = True
                tt[j] = t[j] * np.sqrt(DD)

            else:
                atbd = False
                tt[j] = np.minimum(self.up[j], t[j] * DD)

            ff = self.dace._objfunc(tt)
            nv += 1
            self.perf[:, [nv - 1]] = np.vstack((tt, ff, 2))

            if ff < f:
                t = tt.copy()
                f = ff

            else:
                self.perf[-1, [nv - 1]] = -2

                if not atbd:  # try decrease

                    tt[j] = np.maximum(self.lo[j], t[j] / DD)
                    ff = self.dace._objfunc(tt)
                    nv += 1
                    self.perf[:, [nv - 1]] = np.vstack((tt, ff, 2))

                    if ff < f:
                        t = tt.copy()
                        f = ff

                    else:
                        self.perf[-1, [nv - 1]] = -2

        self.nv = nv

        # other params
        self.f = f
        self.t = t

    def _move(self, th: np.ndarray):
        # pattern move
        nv = self.nv

        t = self.t
        f = self.f

        p = t.size

        v = t / th

        if np.all(v == 1):
            self.D = self.D[np.r_[1:p, 0]] ** 0.2
            return

        # proper move
        rept = True
        while rept:
            tt = np.minimum(self.up, np.maximum(self.lo, t * v))
            ff = self.dace._objfunc(tt)
            nv += 1
            self.perf[:, [nv - 1]] = np.vstack((tt, ff, 3))

            if ff < f:
                t = tt.copy()
                f = ff
                v = v ** 2

            else:
                self.perf[-1, [nv - 1]] = -3
                rept = False

            if np.any(np.logical_or(np.equal(tt, self.lo),
                                    np.equal(tt, self.up))):
                rept = False

        self.nv = nv
        self.D = self.D[np.r_[1:p, 0]] ** 0.25

        # other params
        self.f = f
        self.t = t
