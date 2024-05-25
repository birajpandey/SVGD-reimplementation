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
        num_iterations, step_size = 200, 1e-1

        # Transporter
        transporter = original_svgd.SVGDModel()
        transported = transporter.update(particles, density_obj.score,
                                         n_iter=num_iterations, bandwidth=0.3,
                                         stepsize=step_size, adagrad=True)

        # Plot
        plots.plot_gaussian_mixture_distribution(particles, transported,
                                                 density_obj)


        # check mean
        expected_mean = jnp.sum(weights * means)
        expected_var = (jnp.sum(weights * (covariances.squeeze() + means**2))
                        - expected_mean**2)
        observed_mean = jnp.mean(transported)
        observed_var = jnp.var(transported)

        print(f'Means: Expected={expected_mean} Observed={observed_mean}')
        print(f'Variance: Expected={expected_var} Observed={observed_var}')


        # assert
        # np.testing.assert_array_almost_equal([expected_mean, expected_var],
        #                                      [observed_mean, observed_var],
        #                                      decimal=1,
        #                                      err_msg='Metrics do not match.')

    def test_adagrad_optimizers(self):
        def manual_adagrad_update(theta, grad_theta, historical_grad, stepsize,
                                  alpha=0.9, fudge_factor=1e-6):
            if historical_grad is None:
                historical_grad = np.zeros_like(grad_theta)
            historical_grad = alpha * historical_grad + (1 - alpha) * (
                        grad_theta ** 2)
            adj_grad = grad_theta / (fudge_factor + np.sqrt(historical_grad))
            updated_theta = theta + stepsize * adj_grad
            return updated_theta, historical_grad

        def optax_adagrad_update(theta, grad_theta, optimizer_state,
                                 optimizer):
            updates, optimizer_state = optimizer.update(grad_theta,
                                                        optimizer_state, theta)
            updated_theta = optax.apply_updates(theta, updates)
            return updated_theta, optimizer_state

        np.random.seed(0)
        theta = np.random.randn(10, 5)
        grad_theta = np.random.randn(10, 5)
        stepsize = 1e-2
        alpha = 0.9

        # Manual Adagrad
        historical_grad = np.zeros_like(grad_theta)
        updated_theta_manual, historical_grad = manual_adagrad_update(theta,
                                                                      grad_theta,
                                                                      historical_grad,
                                                                      stepsize,
                                                                      alpha)

        # Optax Adagrad
        optimizer = optax.adagrad(learning_rate=stepsize,
                                  initial_accumulator_value=alpha, eps=1e-6)
        optimizer_state = optimizer.init(theta)
        updated_theta_optax, optimizer_state = optax_adagrad_update(theta,
                                                                    grad_theta,
                                                                    optimizer_state,
                                                                    optimizer)

        # Compare the updates
        np.testing.assert_allclose(updated_theta_manual, updated_theta_optax,
                                   rtol=1e-5, atol=1e-8)
        print("Test passed. Both optimizers produce similar updates.")
