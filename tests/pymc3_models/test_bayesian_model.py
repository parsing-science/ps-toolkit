import unittest

from ps_toolkit.pymc3_models import BayesianModel


class BayesianModelTestCase(unittest.TestCase):
    def test_create_model_raises_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            BM = BayesianModel()
            BM.create_model()

    def test_fit_raises_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            BM = BayesianModel()
            BM.fit()

    def test_predict_proba_raises_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            BM = BayesianModel()
            BM.predict_proba()

    def test_predict_raises_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            BM = BayesianModel()
            BM.predict()

    def test_score_raises_not_implemented_error(self):
        with self.assertRaises(NotImplementedError):
            BM = BayesianModel()
            BM.score()
