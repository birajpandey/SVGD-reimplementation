import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import unittest
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from svgd import kernel, density, models, plots, original_svgd
import optax

class TestModels_w_Original(unittest.TestCase):
    def test_gradients_with_original_implementation(self):
        # define the density
        density_params = {'mean': jnp.array([[-2.0], [2.0]]),
                          'covariance': jnp.array([[[1]], [[1]]]),
                          'weights': jnp.array([1 / 3, 2 / 3])
                          }
        density_obj = density.Density(density.gaussian_mixture_pdf,
                                      density_params)

        # define the particles
        key = jrandom.PRNGKey(10)
        particles = jrandom.normal(key, (5, 1))

        # pick same model
        length_scale = 1.0

        # gradient using original implementation
        orig_svgd = original_svgd.SVGDModel()
        orig_gradient = orig_svgd.calculate_gradients(particles,
                                                      density_obj.score,
                                                      h=length_scale)

        # gradient using new implementation
        model_params = {'length_scale': length_scale}
        model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
        new_svgd = models.SVGDModel(model_kernel)
        new_gradient = new_svgd.calculate_gradient(density_obj, particles)

        # assert
        np.testing.assert_array_almost_equal(orig_gradient,
                                             new_gradient,
                                             decimal=1,
                                             err_msg='Metrics do not match.')

    def test_original_svgd_gaussian_shift_mean(self):
        # define particles
        key = jrandom.PRNGKey(10)
        particles = jrandom.normal(key=key, shape=(500, 1))


        # define 1D density
        density_params = {'mean': jnp.array([10.0]),
            'covariance': jnp.array([[1]])}
        density_obj = density.Density(density.gaussian_pdf, density_params)

        # transport model
        num_iterations, step_size = 500, 1e-2

        # Transporter
        transporter = original_svgd.SVGDModel()
        transported = transporter.update(particles, density_obj.score,
                                         n_iter=num_iterations, bandwidth=50,
                                         stepsize=step_size, adagrad=False)

        # Plot
        plots.plot_distributions(particles, transported, density_params)


    def test_original_svgd_gaussian_dilate_variance(self):
        # define particles
        key = jrandom.PRNGKey(10)
        particles = jrandom.normal(key=key, shape=(500, 1))


        # define 1D density
        density_params = {'mean': jnp.array([4.0]),
            'covariance': jnp.array([[0.3]])}
        density_obj = density.Density(density.gaussian_pdf, density_params)

        # transport model
        num_iterations, step_size = 200, 0.1

        # Transporter
        transporter = original_svgd.SVGDModel()
        transported = transporter.update(particles, density_obj.score,
                                         n_iter=num_iterations, bandwidth=5,
                                         stepsize=step_size, adagrad=False)

        # Plot
        plots.plot_distributions(particles, transported, density_params)


    def test_original_svgd_gaussian_mixture(self):
        # define particles
        key = jrandom.PRNGKey(10)
        particles = jrandom.normal(key=key, shape=(500, 1)) - 10


        # define 1D mixture density
        means = jnp.array([[-2.0], [2.0]])
        covariances = jnp.array([[[1]], [[1]]])
        weights = jnp.array([1/3, 2/3])
        density_params = {'mean': means, 'covariance': covariances,
                          'weights': weights}
        density_obj = density.Density(density.gaussian_mixture_pdf,
                                      density_params)

        # transport model
        num_iterations, step_size = 200, 2.5

        # Transporter
        transporter = original_svgd.SVGDModel()
        transported = transporter.update(particles, density_obj.score,
                                         n_iter=num_iterations, bandwidth=0.3,
                                         stepsize=step_size, adagrad=False)

        # Plot
        plots.plot_gaussian_mixture_distribution(particles, transported,
                                                 density_obj)
