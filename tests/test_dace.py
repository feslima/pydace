import unittest
import numpy as np
from pydace import dacefit, predictor
from .csv_data.csv_read import get_training_data, get_validation_data, get_prediction_data


class TestDace(unittest.TestCase):

    def setUp(self):
        # load the csv data
        self.input_train, self.output_train = get_training_data()
        self.input_val, self.output_val = get_validation_data()
        self.matlab_prediction = get_prediction_data()

    def test_prediction(self):
        MVB, CVB, MVval, CVval = self.input_train, self.output_train, self.input_val, self.output_val

        theta0 = 1 * np.ones((1, 2))
        lob = 1e-3 * np.ones(theta0.shape)
        upb = 100 * np.ones(theta0.shape)

        regr = 'poly0'
        corr = 'corrgauss'

        krmodel = []
        perf = []
        Yhat = np.empty(CVval.shape)
        for i in range(CVB.shape[1]):
            krmodelPH, perfPH = dacefit(MVB, CVB[:, i], regr, corr, theta0, lob, upb)
            krmodel.append(krmodelPH)
            perf.append(perfPH)
            Yhat[:, [i]] = predictor(MVval, krmodelPH)[0]

            # test for closeness of individual values. Discrepancies might be caused by scipy/numpy solvers
            self.assertTrue(np.allclose(Yhat[:, [i]], self.matlab_prediction[:, [i]]))


if __name__ == '__main__':
    unittest.main()
