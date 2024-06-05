import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import jax.numpy as jnp

def plot_distributions(initial_particles, transported_particles, density_params):
    fig = plt.figure(figsize=(10, 6))

    # Plot histogram of initial particles
    plt.hist(initial_particles, bins=30, density=True, alpha=0.4, color='b',
             histtype='bar', label='Initial Particles')

    # Plot histogram of transported particles
    plt.hist(transported_particles, bins=30, density=True, alpha=0.4,
             color='g', histtype='bar', label='Transported Particles')

    # Plot the target density function
    mean = density_params['mean'][0]
    std_dev = np.sqrt(density_params['covariance'][0, 0])
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
    y = norm.pdf(x, mean, std_dev)
    plt.plot(x, y, 'r-', lw=2, label='Target Distribution')

    # plt.title('Initial and Final Distributions of Particles')
    plt.xlabel('Particle Value', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return fig


# Plotting function
def plot_gaussian_mixture_distribution(initial_particles,
                                       transported_particles, density):
    fig = plt.figure(figsize=(10, 6))

    plt.hist(initial_particles, bins=30, density=True, alpha=0.4, color='b', label='Initial Particles')
    plt.hist(transported_particles, bins=30, density=True, alpha=0.4, color='g', label='Transported Particles')


    # plot density object
    means = density.params['mean']
    cov = density.params['mean']
    weights = density.params['weights']
    expected_mean = jnp.sum(weights * means)
    expected_var = (jnp.sum(weights * (cov.squeeze() + means ** 2))
                    - expected_mean ** 2)
    expected_std = jnp.sqrt(expected_var)


    # Plot the target density function
    x = np.linspace(expected_mean - 4 * expected_std,
                    expected_mean + 4 * expected_std, 1000)
    y = density(x)
    plt.plot(x, y, 'r-', lw=2, label='Target Distribution')

    # plt.title('Initial and Final Distributions of Particles')
    plt.xlabel('Particle Value', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.legend(fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return fig


def plot_2d_trajectories(ax, trajectory, num_points, seed=20, alpha=1):
    np.random.seed(seed)
    if trajectory.shape[1] == num_points:
        idx = np.arange(0, num_points)
    else:
        idx = np.random.randint(0, trajectory.shape[1], num_points)
    for i in idx:
        t = trajectory[:, i, :]
        plt.plot(t[:, 0], t[:, 1], '-', c='r', alpha=alpha)
    plt.scatter(trajectory[-1, idx, 0], trajectory[-1, idx, 1],
                s=20, facecolors='k', edgecolors='k', label='Final')
    plt.scatter(trajectory[0, idx, 0], trajectory[0, idx, 1],
                s=20, facecolors='none', edgecolors='k', label='Initial')
    return ax