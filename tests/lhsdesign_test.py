import numpy as np

from pydace.aux_functions import lhsdesign

lb = np.array([8.5, 0., 102., 0.])
ub = np.array([20., 100., 400., 400.])


lhs = lhsdesign(53, lb, ub, include_vertices=False)