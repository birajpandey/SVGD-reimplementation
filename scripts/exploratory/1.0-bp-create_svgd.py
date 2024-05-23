import jax.numpy as jnp
import jax.random as jrandom
from svgd import density, kernel


# generate data
n_samples, n_features = 1000, 1
key = jrandom.PRNGKey(20)
k1, k2, key = jrandom.split(key, 3)
x = jrandom.normal(k1, shape=(n_samples, n_features))

# pick kernel
model_params = {'length_scale': 1.0}
model_kernel = kernel.Kernel(kernel.rbf_kernel, model_params)

# pick gaussian density
means = jrandom.normal(k2, shape=(n_features,))
covariance = jnp.identity(n_features)
density_params = {'mean': means,  'covariance': covariance}
target_density = density.Density(density.gaussian_pdf, density_params)

# check the size of the gram matrix (n_samples, n_samples)
gram_matrix = model_kernel(x, x)
assert gram_matrix.shape == (n_samples, n_samples), \
    ("Gram matrix shape is incorrect.")

# check the size of the score matrix (should be n_samples, n_features)
score_matrix = target_density.score(x)
assert score_matrix.shape == (n_samples, n_features), \
    ("Score matrix shape is incorrect.")

# check the size of the kernel gradient (n_samples, n_samples, n_features)
kernel_gradient = model_kernel.gradient_wrt_first_arg(x, x)
assert kernel_gradient.shape == (n_samples, n_samples, n_features), \
    ("Kernel gradient matrix shape is incorrect")


# calculate the svgd update
kernel_score = jnp.einsum('ij,jk->ik', gram_matrix, score_matrix)
print(f'Size of first term: {kernel_score.shape}' )
second_term = jnp.sum(kernel_gradient, axis=1)
print(f'Size of second term: {second_term.shape}' )
