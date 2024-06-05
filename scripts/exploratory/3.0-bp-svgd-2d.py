import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

import unittest
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from svgd import kernel, density, models, plots, config
import matplotlib.pyplot as plt




class three_gaussian_transport():
    def __init__(self):
        self.reports_dir = str(config.REPORTS_DIR)
        self.means = jnp.array([[-3,0], [3,0], [0, 3]])
        self.covariances = jnp.array([[[0.2, 0],[0, 0.2]], [[0.2, 0],[0, 0.2]],
                                 [[0.2, 0],[0, 0.2]]])
        self.weights = jnp.array([1 / 3, 1 / 3, 1 / 3])
        self.density_params = {'mean': self.means,
                               'covariance': self.covariances,
                               'weights': self.weights}
        self.density_obj = density.Density(density.gaussian_mixture_pdf,
                                      self.density_params)


    def normal_transport(self):
        print('Running normal transport...')
        # generate 2d example
        key = jrandom.PRNGKey(10)
        particles = jrandom.normal(key=key, shape=(500, 2))  * 0.5

        # define model
        model_params = {'length_scale': 0.3}
        model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
        transporter = models.SVGDModel(kernel=model_kernel)


        # transport
        num_iterations, step_size = 1000, 0.5
        transported, trajectory = transporter.predict(particles,
                                                      self.density_obj.score,
                                                      num_iterations, step_size,
                                                      trajectory=True,
                                                      adapt_length_scale=False)


        # plot density
        grid_res = 100
        # Input locations at which to compute log-probabilities
        x_plot = np.linspace(-5, 5, grid_res)
        x_plot = np.stack(np.meshgrid(x_plot, x_plot), axis=-1)

        # Compute log-probabilities
        log_prob = np.log(self.density_obj(x_plot))

        # Reshape to 3D and 2D arrays for plotting
        x_plot = np.reshape(x_plot, (grid_res, grid_res, 2))
        log_prob = np.reshape(log_prob, (grid_res, grid_res))

        # Contourplot levels corresponding to standard deviations
        levels = np.max(np.exp(log_prob)) * np.exp(- np.linspace(4, 0, 5) ** 2)

        # Plot density
        plt.figure(figsize=(5, 5))
        plt.contourf( x_plot[:, :, 0], x_plot[:, :, 1], np.exp(log_prob), cmap="magma")
        plt.scatter(particles[:, 0], particles[:, 1], zorder=2, c='w', s=10,
                    alpha=0.5, label='Initial')
        plt.scatter(transported[:, 0], transported[:, 1], zorder=2, c='r', s=10,
                    alpha=1, label='Final')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.show()


        #  plot trajectory
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax = plots.plot_2d_trajectories(ax, trajectory[::200, :, :], 30,
                                        seed=20)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.show()


    def bad_initialization(self):
        print('Running bad initialization experiment...')
        # generate 2d example
        key = jrandom.PRNGKey(10)
        particles = (jrandom.normal(key=key, shape=(500, 2))  * 0.5 +
                     jnp.array([0, 3]))

        # define model
        model_params = {'length_scale': 0.3}
        model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
        transporter = models.SVGDModel(kernel=model_kernel)


        # transport
        num_iterations, step_size = 1000, 0.5
        transported, trajectory = transporter.predict(particles,
                                                      self.density_obj.score,
                                                      num_iterations, step_size,
                                                      trajectory=True,
                                                      adapt_length_scale=False)


        # plot density
        grid_res = 100
        # Input locations at which to compute log-probabilities
        x_plot = np.linspace(-5, 5, grid_res)
        x_plot = np.stack(np.meshgrid(x_plot, x_plot), axis=-1)

        # Compute log-probabilities
        log_prob = np.log(self.density_obj(x_plot))

        # Reshape to 3D and 2D arrays for plotting
        x_plot = np.reshape(x_plot, (grid_res, grid_res, 2))
        log_prob = np.reshape(log_prob, (grid_res, grid_res))

        # Contourplot levels corresponding to standard deviations
        levels = np.max(np.exp(log_prob)) * np.exp(- np.linspace(4, 0, 5) ** 2)

        # Plot density
        plt.figure(figsize=(5, 5))
        plt.contourf( x_plot[:, :, 0], x_plot[:, :, 1], np.exp(log_prob), cmap="magma")
        plt.scatter(particles[:, 0], particles[:, 1], zorder=2, c='w', s=10,
                    alpha=0.5, label='Initial')
        plt.scatter(transported[:, 0], transported[:, 1], zorder=2, c='r', s=10,
                    alpha=1, label='Final')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.show()

    def strange_circle_transport(self):
        print('Running strange circle transport experiment...')
        # generate 2d example
        key = jrandom.PRNGKey(10)
        particles = jrandom.normal(key=key, shape=(500, 2))  * 0.5 + jnp.array([0, 3])

        # define model
        model_params = {'length_scale': 1.95}
        model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
        transporter = models.SVGDModel(kernel=model_kernel)


        # transport
        num_iterations, step_size = 1000, 0.5
        transported, trajectory = transporter.predict(particles,
                                                      self.density_obj.score,
                                                      num_iterations, step_size,
                                                      trajectory=True,
                                                      adapt_length_scale=False)


        # plot density
        grid_res = 100
        # Input locations at which to compute log-probabilities
        x_plot = np.linspace(-5, 5, grid_res)
        x_plot = np.stack(np.meshgrid(x_plot, x_plot), axis=-1)

        # Compute log-probabilities
        log_prob = np.log(self.density_obj(x_plot))

        # Reshape to 3D and 2D arrays for plotting
        x_plot = np.reshape(x_plot, (grid_res, grid_res, 2))
        log_prob = np.reshape(log_prob, (grid_res, grid_res))

        # Contourplot levels corresponding to standard deviations
        levels = np.max(np.exp(log_prob)) * np.exp(- np.linspace(4, 0, 5) ** 2)

        # Plot density
        plt.figure(figsize=(5, 5))
        plt.contourf( x_plot[:, :, 0], x_plot[:, :, 1], np.exp(log_prob), cmap="magma")
        plt.scatter(particles[:, 0], particles[:, 1], zorder=2, c='w', s=10,
                    alpha=0.5, label='Initial')
        plt.scatter(transported[:, 0], transported[:, 1], zorder=2, c='r', s=10,
                    alpha=1, label='Final')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.show()


if __name__ == '__main__':
    transport_example = three_gaussian_transport()
    transport_example.normal_transport()
    transport_example.bad_initialization()
    transport_example.strange_circle_transport()