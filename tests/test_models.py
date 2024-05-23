import unittest
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from svgd import kernel, density, models, plots


class TestModels(unittest.TestCase):
    def setUp(self):
        return None

    def test_1d_gaussian_test_shift_mean(self):
        # define model
        model_params = {'length_scale': 20.0}
        model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
        transporter = models.SVGDModel(kernel=model_kernel)


        # define particles
        key = jrandom.PRNGKey(10)
        particles = jrandom.normal(key=key, shape=(500, 1))


        # define 1D density
        density_params = {'mean': jnp.array([30.0]),
            'covariance': jnp.array([[1.0]])}
        density_obj = density.Density(density.gaussian_pdf, density_params)

        # transport model
        num_iterations, step_size = 1000, 1e-2
        transported, trajectory = transporter.predict(particles, density_obj,
                                               step_size, num_iterations,
                                               trajectory=True)

        # check mean
        expected_mean = density_params['mean'][0]
        expected_var = density_params['covariance'][0, 0]
        observed_mean = jnp.mean(transported)
        observed_var = jnp.var(transported)

        print(transported.shape, trajectory.shape)
        print(f'Means: Expected={expected_mean} Observed={observed_mean}')
        print(f'Variance: Expected={expected_var} Observed={observed_var}')

        # plot
        plots.plot_distributions(particles, trajectory[-1], density_params)

        # assert
        np.testing.assert_almost_equal(expected_mean, observed_mean, decimal=1,
                 err_msg='Means do not match.')
        np.testing.assert_almost_equal(expected_var, observed_var, decimal=1,
                 err_msg='Variances do not match.')

    def test_1d_gaussian_dilate_variance(self):
        # define model
        model_params = {'length_scale': 10.0}
        model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
        transporter = models.SVGDModel(kernel=model_kernel)


        # define particles
        key = jrandom.PRNGKey(10)
        particles = jrandom.normal(key=key, shape=(500, 1))


        # define 1D density
        density_params = {'mean': jnp.array([10.0]),
            'covariance': jnp.array([[0.1]])}
        density_obj = density.Density(density.gaussian_pdf, density_params)

        # transport model
        num_iterations, step_size = 1000, 1e-2
        transported, trajectory = transporter.predict(particles, density_obj,
                                               step_size, num_iterations,
                                               trajectory=True)

        # check mean
        expected_mean = density_params['mean'][0]
        expected_var = density_params['covariance'][0, 0]
        observed_mean = jnp.mean(transported)
        observed_var = jnp.var(transported)

        print(transported.shape, trajectory.shape)
        print(f'Means: Expected={expected_mean} Observed={observed_mean}')
        print(f'Variance: Expected={expected_var} Observed={observed_var}')

        # plot
        plots.plot_distributions(particles, trajectory[-1], density_params)

        # assert
        np.testing.assert_almost_equal(expected_mean, observed_mean, decimal=1,
                 err_msg='Means do not match.')
        np.testing.assert_almost_equal(expected_var, observed_var, decimal=1,
                 err_msg='Variances do not match.')





if __name__ == '__main__':
    unittest.main()

