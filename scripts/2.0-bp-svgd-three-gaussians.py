import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

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

        # Plot density
        grid_res = 100
        # Input locations at which to compute probabilities
        x_plot = np.linspace(-4.5, 4.5, grid_res)
        x_plot = np.stack(np.meshgrid(x_plot, x_plot), axis=-1)

        # Plot density
        prob = self.density_obj(x_plot.reshape(-1, 2)).reshape(grid_res,
                                                               grid_res)
        plt.figure(figsize=(5, 5))
        plt.contourf(x_plot[:, :, 0], x_plot[:, :, 1],
                     prob, cmap="magma")

        # plot initial particles
        plt.scatter(particles[:, 0], particles[:, 1], zorder=2, c="w", s=10,
                    label="initial sample", alpha=0.5)

        # plot final particles
        plt.scatter(transported[:, 0], transported[:, 1], zorder=2, c='r',
                    s=10, label="final sample", alpha=0.6)
        plt.xlim(-4.5, 4.5)
        plt.ylim(-4.5, 4.5)
        plt.legend()
        plt.title(f"Transported Particles, h={model_params['length_scale']}")
        plt.savefig(self.reports_dir + '/figures/3_gaussians.pdf',
                    dpi=600)
        plt.show()


        #  plot trajectory
        n = 500
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax = plots.plot_2d_trajectories(ax, trajectory, n, seed=20,
                                        alpha=0.1)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        plt.legend()
        ax.set_title(f"Trajectory of {n} particles")
        plt.savefig(self.reports_dir + '/figures/3_gaussians_trajectory.pdf',
                    dpi=600)
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


        # Plot density
        grid_res = 100
        # Input locations at which to compute probabilities
        x_plot = np.linspace(-4.5, 4.5, grid_res)
        x_plot = np.stack(np.meshgrid(x_plot, x_plot), axis=-1)

        # Plot density
        prob = self.density_obj(x_plot.reshape(-1, 2)).reshape(grid_res,
                                                               grid_res)
        fig = plt.figure(figsize=(5, 5))
        plt.contourf(x_plot[:, :, 0], x_plot[:, :, 1],
                     prob, cmap="magma")

        # plot initial particles
        plt.scatter(particles[:, 0], particles[:, 1], zorder=2, c="w", s=10,
                    label="initial sample", alpha=0.5)

        # plot final particles
        plt.scatter(transported[:, 0], transported[:, 1], zorder=2, c='r',
                    s=10, label="final sample", alpha=0.6)
        plt.xlim(-4.5, 4.5)
        plt.ylim(-4.5, 4.5)
        plt.legend()
        plt.title(f"Transported Particles, h={model_params['length_scale']}")
        plt.savefig(self.reports_dir +
                    '/figures/3_gaussians_bad_init.pdf',
                    dpi=600)
        plt.show()


        #  plot trajectory
        n = 500
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax = plots.plot_2d_trajectories(ax, trajectory, n, seed=20,
                                        alpha=0.1)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        plt.legend()
        ax.set_title(f"Trajectory of {n} particles")
        plt.savefig(self.reports_dir +
                    '/figures/3_gaussians_bad_init_trajectory.pdf',
                    dpi=600)
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


        # Plot density
        grid_res = 100
        # Input locations at which to compute probabilities
        x_plot = np.linspace(-4.5, 4.5, grid_res)
        x_plot = np.stack(np.meshgrid(x_plot, x_plot), axis=-1)

        # Plot density
        prob = self.density_obj(x_plot.reshape(-1, 2)).reshape(grid_res,
                                                               grid_res)
        fig = plt.figure(figsize=(5, 5))
        plt.contourf(x_plot[:, :, 0], x_plot[:, :, 1],
                     prob, cmap="magma")

        # plot initial particles
        plt.scatter(particles[:, 0], particles[:, 1], zorder=2, c="w", s=10,
                    label="initial sample", alpha=0.5)

        # plot final particles
        plt.scatter(transported[:, 0], transported[:, 1], zorder=2, c='r',
                    s=10, label="final sample", alpha=0.6)
        plt.xlim(-4.5, 4.5)
        plt.ylim(-4.5, 4.5)
        plt.title(f"Transported Particles, h={model_params['length_scale']}")
        plt.legend()
        plt.savefig(self.reports_dir +
                    '/figures/3_gaussians_strange_circle.pdf',
                    dpi=600)
        plt.show()

        #  plot trajectory
        n = 500
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax = plots.plot_2d_trajectories(ax, trajectory, n, seed=20,
                                        alpha=0.03)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        plt.legend()
        ax.set_title(f"Trajectory of {n} particles")
        plt.savefig(self.reports_dir +
                    '/figures/3_gaussians_strange_circle_trajectory.pdf',
                    dpi=600)
        plt.show()


if __name__ == '__main__':
    transport_example = three_gaussian_transport()
    transport_example.normal_transport()
    transport_example.bad_initialization()
    transport_example.strange_circle_transport()