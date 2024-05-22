import unittest
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from svgd.density import Density, gaussian_pdf

class TestDensity(unittest.TestCase):

    def setUp(self):
        # Parameters for the Gaussian distribution
        self.params = {
            'mean': jnp.array([0.0]),
            'covariance': jnp.array([[1.0]])
        }
        # Create the Density object with the Gaussian PDF
        self.density_obj = Density(gaussian_pdf, self.params)

    def test_density_single_point(self):
        point = jnp.array([0.0])
        density_value = self.density_obj.density(point)
        expected_value = multivariate_normal.pdf(point, self.params['mean'], self.params['covariance'])
        self.assertAlmostEqual(density_value, expected_value, places=5)

    def test_density_multiple_points(self):
        points = jnp.array([[0.0], [1.0], [2.0]])
        density_values = self.density_obj(points)
        expected_values = jax.vmap(lambda x: multivariate_normal.pdf(x, self.params['mean'], self.params['covariance']))(points)
        for val, exp_val in zip(density_values, expected_values):
            self.assertAlmostEqual(val, exp_val, places=5)

    def test_score_single_point(self):
        point = jnp.array([0.0])
        score_value = self.density_obj.score(point)
        log_density = lambda x: jnp.log(multivariate_normal.pdf(x, self.params['mean'], self.params['covariance']))
        expected_score = jax.grad(log_density)(point)
        self.assertTrue(jnp.isclose(score_value, expected_score, rtol=1e-5),
        "Score values for single point is incorrect.")

    def test_score_multiple_points(self):
        points = jnp.array([[0.0], [1.0], [2.0]])
        score_values = self.density_obj.score(points)
        log_density = lambda x: jnp.log(multivariate_normal.pdf(x, self.params['mean'], self.params['covariance']))
        expected_scores = jax.vmap(jax.grad(log_density))(points)
        self.assertTrue(jnp.allclose(score_values, expected_scores, rtol=1e-5),
        "Score values for multiple points is incorrect.")

if __name__ == '__main__':
    unittest.main()
