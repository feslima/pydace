import scipy.io as sio
import numpy as np
from pydace import dacefit, predictor
import matplotlib.pyplot as plt
import time


mat_contents = sio.loadmat('doe_final_infill.mat')

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
    krmodelPH, perfPH = dacefit(MVB, CVB[:, i], 'poly1', 'corrgauss', theta0, lob, upb)
    krmodel.append(krmodelPH)
    perf.append(perfPH)
    Yhat[:, [i]] = predictor(MVval, krmodelPH)[0]

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