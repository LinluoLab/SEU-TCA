import unittest
import numpy as np
from seu_tca import SEU_TCA

class TestSEUTCA(unittest.TestCase):
    def test_fit_predict(self):
        Xs = np.random.rand(100, 50)
        Xt = np.random.rand(100, 50)
        Ys = np.random.randint(0, 2, size=(100, 1))
        Yt = np.random.randint(0, 2, size=(100, 1))

        model = SEU_TCA(kernel_type='rbf', dim=10, gamma=0.1)
        acc, _ = model.fit_predict(Xs, Ys, Xt, Yt)
        self.assertTrue(0 <= acc <= 1)

if __name__ == "__main__":
    unittest.main()