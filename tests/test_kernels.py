from re import X
import unittest
import jax 
import jax.numpy as jnp
import jax.random as jrandom
from svgd.kernel import Kernel, rbf_kernel
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel

class TestKernel(unittest.TestCase):

  def setUp(self):
    self.params = {'length_scale': 1.0}
    self.kernel = Kernel(rbf_kernel, self.params)

    # generate data
    n_samples, n_features = 5, 4
    key = jrandom.PRNGKey(20)
    k1, k2, key = jrandom.split(key, 3)
    self.x = jrandom.normal(k1, shape=(n_samples, n_features))
    self.y = jrandom.normal(k2, shape=(n_samples, n_features))

  def test_rbf_kernel(self):
    kernel_matrix = self.kernel(self.x, self.y)
    expected_matrix = expected_value = sklearn_rbf_kernel(self.x, self.y, gamma=1.0 / (2 * self.params['length_scale'] ** 2))
    self.assertTrue(jnp.allclose(kernel_matrix, expected_matrix),
    "Kernel matrix for rbf kernel is not the same.")



  if __name__ == 'main':
    unittest.main()


