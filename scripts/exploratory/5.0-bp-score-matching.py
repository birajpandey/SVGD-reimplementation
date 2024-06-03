import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom



# define the dataset
# Need a way to generate samples
# Need analytical scores for comparison


# define the

# define the model
key = jrandom.PRNGKey(20)
model_key, key = key.split(2)
model = eqx.nn.MLP(in_size=2, out_size=2, width_size=16, depth=2,
                   key=model_key)

# define the loss
def score_matching_loss(y_true, y_pred):
    return None