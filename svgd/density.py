import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
import equinox as eqx

class Density(eqx.Module):
    """ General Density class for a given probability density function.
    """
    pdf_fun: callable
    params: dict

    def __init__(self, pdf_fun, params):
        self.pdf_fun = pdf_fun
        self.params = params

    def density(self, x):
        return self.pdf_fun(x, self.params)

    def __call__(self, x):
        return jax.vmap(self.density)(x)

    def score(self, x):
        log_density = lambda x: jnp.log(self.density(x))
        score_fun = jax.grad(log_density, argnums=0)
        return jax.vmap(score_fun)(x)

# Define a Gaussian PDF for testing
def gaussian_pdf(x, params):
    mean = params['mean']
    cov = params['covariance']
    return multivariate_normal.pdf(x, mean, cov)