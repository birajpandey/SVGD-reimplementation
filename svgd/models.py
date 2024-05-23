import jax.numpy as jnp
from copy import deepcopy
import numpy as np
import jax
from tqdm import tqdm
import equinox as eqx


class SVGDModel(eqx.Module):
    kernel_obj: eqx.Module

    def __init__(self, kernel):
        self.kernel_obj = kernel

    @eqx.filter_jit
    def calculate_gradient(self, density, particles):
        num_particles = particles.shape[0]
        gram_matrix = self.kernel_obj(particles, particles)
        score_matrix = density.score(particles)
        kernel_gradient = self.kernel_obj.gradient_wrt_first_arg(particles,
                                                                 particles)

        # calculate the terms
        kernel_score_term = jnp.einsum('ij,jk->ik', gram_matrix, score_matrix)
        kernel_gradient_term = jnp.sum(kernel_gradient, axis=1)
        phi = (kernel_score_term + kernel_gradient_term) / num_particles

        return phi

    @eqx.filter_jit
    def update(self, particles, density, step_size):
        gradient = self.calculate_gradient(density, particles)
        updated_particles = particles + step_size * gradient
        return updated_particles

    def predict(self, particles, density, step_size, num_iterations,
                trajectory=False):
        particle_trajectory = np.zeros((num_iterations + 1, particles.shape[0],
                              particles.shape[1]))
        start = deepcopy(particles)
        particle_trajectory[0] = start

        # append
        for i in tqdm(range(1, num_iterations + 1)):
            updated_particles = self.update(start, density, step_size)
            # print(updated_particles)
            particle_trajectory[i] = updated_particles
            start = updated_particles

        if trajectory:
            return particle_trajectory[-1], particle_trajectory

        else:
            return particle_trajectory[-1]



