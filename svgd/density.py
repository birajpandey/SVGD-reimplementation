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
        log_density = lambda x: jnp.clip(jnp.log(self.density(x)),
                                         a_min=-1e10,
                                         a_max=1e10)
        score_fun = jax.grad(log_density, argnums=0)
        return jax.vmap(score_fun)(x)

# Define a unimodal gaussian pdf
def gaussian_pdf(x, params):
    mean = params['mean']
    cov = params['covariance']
    return multivariate_normal.pdf(x, mean, cov)


# define a bimodal gaussian
def gaussian_mixture_pdf(x, params):
    """
    Compute the probability density of a Gaussian Mixture model.


    Parameters:
        x: jnp.ndarray
            The input data points of shape (num_points, num_features)
        weights: jnp.ndarray
            The mixture weights os shape (num_components,).
        means: jnp.ndarray
            The means of shape (num_components, num_features).
        covs : jnp.ndarray
            The covariances of shape (num_components, num_features,
            num_features).

    Returns:
    jnp.ndarray
        The probability densities for each data point, shape (num_points,)
    """

    # params
    mean = params['mean']
    cov = params['covariance']
    weights = params['weights']
    num_components = weights.shape[0]

    # Compute the PDF of each component
    component_pdfs = jnp.array([multivariate_normal.pdf(x, mean[k], cov[k])
                                for k in range(num_components)])

    # Weighted sum of the component PDFs
    pdf = jnp.dot(weights, component_pdfs)
    return pdf


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

