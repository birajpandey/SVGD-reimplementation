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
    def calculate_gradient(self, score_obj, particles, kernel_params=None):
        num_particles = particles.shape[0]
        gram_matrix = self.kernel_obj(particles, particles, kernel_params)
        score_matrix = score_obj(particles)
        kernel_gradient = self.kernel_obj.gradient_wrt_first_arg(particles,
                                                                 particles,
                                                                 kernel_params)

        # calculate the terms
        kernel_score_term = jnp.einsum('ij,jk->ik', gram_matrix, score_matrix)
        kernel_gradient_term = jnp.sum(kernel_gradient, axis=0)
        phi = (kernel_score_term + kernel_gradient_term) / num_particles
        return phi
    
    @eqx.filter_jit
    def calculate_length_scale(self, particles):
        pairwise_sq_distances= jax.vmap(
            lambda x1: jax.vmap(lambda y1: jnp.sum((x1 - y1) ** 2))(
                particles))(particles)
        median_distance = jnp.sqrt(jnp.median(pairwise_sq_distances))
        new_length_scale = 0.5 * median_distance / jnp.log(len(particles))
        return new_length_scale

    @eqx.filter_jit
    def update(self, particles, score_obj, step_size, kernel_params=None):
        gradient = self.calculate_gradient(score_obj, particles, kernel_params)
        # print(f'Gradient: {jnp.linalg.norm(gradient)}')
        updated_particles = particles + step_size * gradient
        return updated_particles

    def predict(self, particles, score_obj, num_iterations, step_size,
                trajectory=False, adapt_length_scale=False):
        particle_trajectory = np.zeros((num_iterations + 1, particles.shape[0],
                              particles.shape[1]))
        start = deepcopy(particles)
        particle_trajectory[0] = start
        kernel_params = self.kernel_obj.params # initialize

        for i in tqdm(range(1, num_iterations + 1)):
            if 'length_scale' in kernel_params and adapt_length_scale:
                # Update length scale outside of JIT
                kernel_params['length_scale'] = self.calculate_length_scale(start)
            # print(f'Kernel params: {kernel_params}')
            start = self.update(start, score_obj, step_size, kernel_params)
            particle_trajectory[i] = start

        if trajectory:
            return particle_trajectory[-1], particle_trajectory

        else:
            return particle_trajectory[-1]



