
import unittest
import numpy as np
from src.optimizer import HybridOptimizerV39

class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.opt = HybridOptimizerV39()

    def test_entropy_output(self):
        history = np.random.randn(128).tolist()
        entropy = self.opt.get_mps_entropy(history)
        self.assertIsInstance(entropy, (float, np.float64))

if __name__ == '__main__':
    unittest.main()
