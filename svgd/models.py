import jax.numpy as jnp
class SVGDModel:
    def __init__(self, density, kernel):
        self.density_obj = density
        self.kernel_obj = kernel

    def calculate_gradient(self, particles):
        gram_matrix = self.kernel_obj(particles)
        score = self.density_obj.score(particles)()


    def update(self, particles, step_size):
        return None
