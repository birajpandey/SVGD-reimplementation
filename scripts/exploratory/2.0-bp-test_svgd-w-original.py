import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from svgd import kernel, density, models, plots, original_svgd
import optax
from copy import deepcopy

# define the particles
key = jrandom.PRNGKey(0)
particles = jrandom.normal(key, (5, 1))  # Example particles
length_scale = 1.0


## Test the kernel implementation
# Initialize SVGD and compute gradient with the original implementation
orig_svgd = original_svgd.OriginalSVGD()
_, dxkxy_original = orig_svgd.svgd_kernel(particles, h=length_scale)

# Compute gradients using kernel class
model_params  = {'length_scale': length_scale}
model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)
dxkxy_jax = model_kernel.gradient_wrt_first_arg(particles, particles,
                                                model_params)

np.testing.assert_allclose(dxkxy_original, dxkxy_jax.sum(axis=0), rtol=1e-5,
                           atol=1e-8, err_msg='Gradients dont match')


# Test the gradient implementation

# define 1D mixture density
means = jnp.array([[-2.0], [2.0]])
covariances = jnp.array([[[1]], [[1]]])
weights = jnp.array([1 / 3, 2 / 3])
density_params = {'mean': means, 'covariance': covariances,
                  'weights': weights}
density_obj = density.Density(density.gaussian_mixture_pdf, density_params)


# new svgd object
new_svgd = models.SVGDModel(model_kernel)
new_gradient = new_svgd.calculate_gradient(density_obj, particles)

orig_gradient = orig_svgd.calculate_gradients(particles, density_obj.score,
                                              h=length_scale)

np.testing.assert_allclose(new_gradient, orig_gradient, rtol=1e-5,
                           atol=1e-8, err_msg='Gradients dont match')