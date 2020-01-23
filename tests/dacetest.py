import scipy.io as sio
import numpy as np
from pydace import Dace
import matplotlib.pyplot as plt
import time

import pathlib

filepath = str(pathlib.Path(__file__).resolve().parent /
               'doe_final_infill.mat')
mat_contents = sio.loadmat(filepath)

CV = mat_contents['CV']
MV = mat_contents['MV']

theta0 = 10 * np.ones((1, 2))
lob = 1e-3 * np.ones(theta0.shape)
upb = 100 * np.ones(theta0.shape)

CVval = CV[:99, :]
MVval = MV[:99, :]

CVB = CV[100:, :]
MVB = MV[100:, :]

var_labels = ['L/F', 'V/F', 'xD', 'xB', 'J', 'QR']


start = time.time()
krmodel = []
perf = []
Yhat = np.empty(CVval.shape)
for i in np.arange(CV.shape[1]):
    krmodelPH = Dace(regression='poly1', correlation='corrgauss')
    krmodelPH.fit(S=MVB, Y=CVB[:, i], theta0=theta0, lob=lob, upb=upb)
    
    krmodel.append(krmodelPH)
    perf.append(krmodelPH.perf)
    
    Yhat[:, [i]], *_ = krmodelPH.predict(X=MVval)

end = time.time()
exectime = end - start
for var in np.arange(CV.shape[1]):
    plt.figure(var + 1)
    plt.plot(CVval[:, var], Yhat[:,var], 'b+')
    plt.xlabel(var_labels[var] + ' - Validation')
    plt.ylabel(var_labels[var] + ' - Kriging')
    # plt.show()

plt.show()

print(f'Dacefit exec time = {exectime} seconds.')
