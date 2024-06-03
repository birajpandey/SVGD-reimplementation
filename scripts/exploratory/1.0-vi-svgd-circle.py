import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax.numpy as jnp
import jax.random as jrandom
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


class RingDistribution:
    """Two-dimensional probability distribution centered around a ring, with
    radial normal noise and uniform angular distribution.
    """

    def __init__(self, radius, std):
        self.mean = radius
        self.std = std

    def sample(self, n_samples, seed=None):
        """Return an array of samples from the distribution."""
        np.random.seed(seed)
        r = np.random.normal(loc=self.mean, scale=self.std, size=n_samples)
        theta = np.random.uniform(low=-np.pi, high=np.pi, size=n_samples)

        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)
        x = np.array([x1, x2]).T
        return x

    def pdf(self, x):
        """Probability density function."""
        r = np.sqrt((x**2).sum(axis=1))
        return norm.pdf(r, loc=self.mean, scale=self.std)

    def score(self, x):
        """Gradient of the log of the PDF."""
        r = jnp.sqrt((x**2).sum(axis=1))
        scores = x * (RADIUS / r - 1).reshape(-1, 1)
        return scores


RADIUS = 1
STD = 0.1
distribution = RingDistribution(radius=RADIUS, std=STD)

X = distribution.sample(n_samples=10000, seed=1234)

# Create a grid of points for plots.
size = RADIUS + 5 * STD
step = 0.1
x_grid, y_grid = np.meshgrid(
    np.arange(-size, size + step, step), np.arange(-size, size + step, step)
)
grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]

# f, ax = plt.subplots()
# ax.scatter(X[:, 0], X[:, 1], edgecolors="k")

scores = distribution.score(grid_points)
# ax.quiver(x_grid, y_grid, scores[:, 0], scores[:, 1])
# ax.set_title("Samples (dots) and true score function (vectors)")
# plt.show()

######################################################
# generate 2D example
key = jrandom.PRNGKey(10)
particles = jrandom.normal(key=key, shape=(500, 2)) * 0.4 + jnp.array([3,0])

# define model
model_params = {"length_scale": 0.025}
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
plt.title("Transported Particles")

plt.show()

# plot trajectory
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax = plots.plot_2d_trajectories(ax, trajectory, 2000, seed=20, alpha=0.03)
ax.set_xlim(-2, 4)
ax.set_ylim(-3, 3)
ax.set_title("Trajectories of 2000 particles")
plt.show()