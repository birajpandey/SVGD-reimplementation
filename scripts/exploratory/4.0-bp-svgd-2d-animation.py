import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import unittest
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from svgd import kernel, density, models, plots, original_svgd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 150
plt.ioff()



# generate 2D example
key = jrandom.PRNGKey(10)
particles = jrandom.normal(key=key, shape=(500, 2))  * 0.5

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
num_iterations, step_size = 49, 2.5
transported, trajectory = transporter.predict(particles, density_obj,
                                              num_iterations, step_size,
                                              trajectory=True)

# plot trajectory
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax = plots.plot_2d_trajectories(ax, trajectory, 50, seed=20)
plt.show()

# make animation
idx = np.random.randint(0, trajectory.shape[1], 70)
sel_trajectory = trajectory[:, idx, :]

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xticks([])
ax.set_yticks([])

paths = []
start_markers = []
end_markers = []

for i in range(sel_trajectory.shape[1]):
    path, = ax.plot([], [], 'r-', lw=1)
    start_marker, = ax.plot([], [], 'o', mec='black', mfc='None', markersize=5)
    end_marker, = ax.plot([], [], 'o', mec='black', mfc='black', markersize=5)
    paths.append(path)
    start_markers.append(start_marker)
    end_markers.append(end_marker)

def init():
    for path, start_marker, end_marker in zip(paths, start_markers, end_markers):
        path.set_data([], [])
        start_marker.set_data([], [])
        end_marker.set_data([], [])
    return paths + start_markers + end_markers

def update(frame):
    for i, (path, start_marker, end_marker) in enumerate(zip(paths, start_markers, end_markers)):
        x_values = sel_trajectory[:, i, 0]
        y_values = sel_trajectory[:, i, 1]
        path.set_data(x_values[:frame + 1], y_values[:frame + 1])
        if frame == 0:
            start_marker.set_data([sel_trajectory[0, i, 0]], [sel_trajectory[0, i, 1]])
        if frame == sel_trajectory.shape[0] - 1:
            end_marker.set_data([sel_trajectory[-1, i, 0]], [sel_trajectory[-1, i, 1]])
    return paths + start_markers + end_markers

print('Creating animation...')
plt.tight_layout()

ani = FuncAnimation(fig, update, frames=range(sel_trajectory.shape[0]), init_func=init, blit=True)

reports_path = 'reports/figures/'
figure_name = '3_gaussians_trajectory_video.gif'
writergif = PillowWriter(fps=10)
ani.save(reports_path + figure_name, dpi=600, writer=writergif)
print(f'Saved figure {figure_name}')