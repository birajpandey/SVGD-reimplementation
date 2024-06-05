import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import jax.random as jrandom
import jax
import numpy as np
from svgd import kernel, density, models, plots, original_svgd

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Matplotlib parameters
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["image.cmap"] = "RdBu"
bright_colormap = ListedColormap(["#FF0000", "#0000FF"])


RADIUS = 1
STD = 0.1
distribution = density.RingDistribution(radius=RADIUS, std=STD)

######################################################
# generate 2D example
key = jrandom.PRNGKey(10)
n = 500
particles = jrandom.normal(key=key, shape=(n, 2)) * 0.4 + jnp.array([3,0])

# define model
kernel_bandwidth = 0.025
model_params = {"length_scale": kernel_bandwidth}
model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
transporter = models.SVGDModel(kernel=model_kernel)

# transport
num_iterations, step_size = 20000, 2
transported, trajectory = transporter.predict(
    particles, distribution.score, num_iterations, step_size, trajectory=True
)

# plot density
grid_res = 100
# Input locations at which to compute probabilities
x_plot = np.linspace(-4.5, 4.5, grid_res)
x_plot = np.stack(np.meshgrid(x_plot, x_plot), axis=-1)

# Plot density
prob = distribution.pdf(x_plot.reshape(-1, 2)).reshape(grid_res, grid_res)
plt.figure(figsize=(5, 5))
plt.contourf(x_plot[:, :, 0], x_plot[:, :, 1], jnp.cbrt(jnp.cbrt(prob)), cmap="magma")

# plot initial particles
plt.scatter(particles[:, 0], particles[:, 1], zorder=2, c="w", s=10, label="initial sample", alpha=0.5)

# plot final particles
plt.scatter(transported[:, 0], transported[:, 1], zorder=2, c='r', s=10, label="final sample", alpha=0.6)
plt.legend()
plt.title(f"Transported Particles, h={kernel_bandwidth}")

plt.show()

# plot trajectory
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax = plots.plot_2d_trajectories(ax, trajectory, n, seed=20, alpha=0.03)
ax.set_xlim(-2, 4)
ax.set_ylim(-3, 3)
ax.set_title(f"Trajectories of {n} particles")
plt.show()