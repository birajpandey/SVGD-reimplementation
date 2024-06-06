import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from svgd import kernel, density, models, plots, config


class one_dim_transport():
    def gaussian_test_shift_mean(self):
        print("Running gaussian shift:")
        # define model
        model_params = {'length_scale': 50.0}
        model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
        transporter = models.SVGDModel(kernel=model_kernel)


        # define particles
        key = jrandom.PRNGKey(10)
        particles = jrandom.normal(key=key, shape=(700, 1))


        # define 1D density
        density_params = {'mean': jnp.array([10.0]),
            'covariance': jnp.array([[1.0]])}
        density_obj = density.Density(density.gaussian_pdf, density_params)

        # transport model
        num_iterations, step_size = 1000, 1e-2
        transported, trajectory = transporter.predict(particles, density_obj.score,
                                               num_iterations, step_size,
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
        fig = plots.plot_distributions(particles, trajectory[-1], density_params)
        fig.savefig(str(config.REPORTS_DIR) +
                    '/figures/1d_gaussian_test_shift_mean.pdf', dpi=600)


    def gaussian_dilate_variance(self):
        print("Running gaussian dilate:")
        model_params = {'length_scale': 5.0}
        model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
        transporter = models.SVGDModel(kernel=model_kernel)


        # define particles
        key = jrandom.PRNGKey(10)
        particles = jrandom.normal(key=key, shape=(700, 1))


        # define 1D density
        density_params = {'mean': jnp.array([4.0]),
            'covariance': jnp.array([[0.3]])}
        density_obj = density.Density(density.gaussian_pdf,
                                      density_params)

        # transport model
        num_iterations, step_size = 1000, 0.1
        transported, trajectory = transporter.predict(particles, density_obj.score,
                                               num_iterations, step_size,
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
        fig = plots.plot_distributions(particles, trajectory[-1],
                                      density_params)
        fig.savefig(str(config.REPORTS_DIR) +
                    '/figures/1d_gaussian_dilate_variance.pdf', dpi=600)


    def gaussian_mixture(self):
        print("Running gaussian mixture:")
        # define particles
        key = jrandom.PRNGKey(10)
        particles = jrandom.normal(key=key, shape=(5000, 1)) - 10

        # define model
        model_params = {'length_scale': 0.65}
        model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
        transporter = models.SVGDModel(kernel=model_kernel)

        # define 1D mixture density
        means = jnp.array([[-2.0], [2.0]])
        covariances = jnp.array([[[1]], [[1]]])
        weights = jnp.array([1/3, 2/3])
        density_params = {'mean': means, 'covariance': covariances,
                          'weights': weights}
        density_obj = density.Density(density.gaussian_mixture_pdf,
                                      density_params)

        # transport model
        num_iterations, step_size = 500, 3

        # define the optimizer
        transported, trajectory = transporter.predict(particles, density_obj.score,
                                               num_iterations, step_size,
                                                      trajectory=True)

        # check mean
        expected_mean = jnp.sum(weights * means.flatten())
        expected_var = (jnp.sum(weights * (covariances.squeeze() +
                                           means.flatten()**2))
                        - expected_mean**2)
        observed_mean = jnp.mean(transported)
        observed_var = jnp.var(transported)

        print(f'Means: Expected={expected_mean} Observed={observed_mean}')
        print(f'Variance: Expected={expected_var} Observed={observed_var}')

        # plot
        fig = plots.plot_gaussian_mixture_distribution(particles, trajectory[
            -1], density_obj)
        fig.savefig(str(config.REPORTS_DIR) +
                    '/figures/1d_gaussian_mixture.pdf', dpi=600)



if __name__ == '__main__':
    transport_experiment = one_dim_transport()
    transport_experiment.gaussian_test_shift_mean()
    transport_experiment.gaussian_dilate_variance()
    transport_experiment.gaussian_mixture()

