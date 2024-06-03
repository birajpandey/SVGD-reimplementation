import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

import unittest
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from svgd import kernel, density, models, plots, original_svgd
import matplotlib.pyplot as plt

# generate 2D example
key = jrandom.PRNGKey(10)
particles = jrandom.normal(key=key, shape=(500, 2))  * 0.5 + jnp.array([0, 3])

# define model
model_params = {'length_scale': 0.3}
model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
transporter = models.SVGDModel(kernel=model_kernel)

# define 1D mixture density
means = jnp.array([[-3,0], [3,0], [0, 3]])
covariances = jnp.array([[[0.2, 0],[0, 0.2]], [[0.2, 0],[0, 0.2]],
                         [[0.2, 0],[0, 0.2]]])
weights = jnp.array([1 / 3, 1 / 3, 1 / 3])
density_params = {'mean': means, 'covariance': covariances,
                  'weights': weights}
density_obj = density.Density(density.gaussian_mixture_pdf,
                              density_params)


# transport
num_iterations, step_size = 100, 0.5
transported, trajectory = transporter.predict(particles, density_obj.score,
                                              num_iterations, step_size,
                                              trajectory=True)


# plot density
grid_res = 100
# Input locations at which to compute log-probabilities
x_plot = np.linspace(-5, 5, grid_res)
x_plot = np.stack(np.meshgrid(x_plot, x_plot), axis=-1)

# Compute log-probabilities
log_prob = np.log(density_obj(x_plot))

# Reshape to 3D and 2D arrays for plotting
x_plot = np.reshape(x_plot, (grid_res, grid_res, 2))
log_prob = np.reshape(log_prob, (grid_res, grid_res))

# Contourplot levels corresponding to standard deviations
levels = np.max(np.exp(log_prob)) * np.exp(- np.linspace(4, 0, 5) ** 2)

# Plot density
plt.figure(figsize=(5, 5))
plt.contourf( x_plot[:, :, 0], x_plot[:, :, 1], np.exp(log_prob),
              cmap="magma")

# plot initial particles
plt.scatter(particles[:, 0], particles[:, 1], zorder=2, c='w', s=10)


# plot final particles
# plt.scatter(transported[:, 0], transported[:, 1], zorder=2, c='r', s=10)

plt.show()
