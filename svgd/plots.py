import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_distributions(initial_particles, transported_particles, density_params):
    plt.figure(figsize=(10, 6))

    # Plot histogram of initial particles
    plt.hist(initial_particles, bins=30, density=True, alpha=0.4, color='b', label='Initial Particles')

    # Plot histogram of transported particles
    plt.hist(transported_particles, bins=30, density=True, alpha=0.4, color='g', label='Transported Particles')

    # Plot the target density function
    mean = density_params['mean'][0]
    std_dev = np.sqrt(density_params['covariance'][0, 0])
    x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 1000)
    y = norm.pdf(x, mean, std_dev)
    plt.plot(x, y, 'r-', lw=2, label='Target Distribution')

    plt.title('Initial and Final Distributions of Particles')
    plt.xlabel('Particle Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()