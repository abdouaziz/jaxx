import jax 
import jax.numpy as jnp
from jax import jit , grad , vmap , random




key = random.PRNGKey(seed=100)

rand = random.randint(key, shape=(10, 10), minval=0, maxval=10)

print(rand)