class HopfieldNetwork(eqx.Module):
  """ A simple Hopfield Network (we use ReLU as the activation function) replaces the MLP in traditional Transformers """
  Xi: jax.Array

  def __init__(self, key:jr.PRNGKey, config:ETConfig):
    nmems = int(config.scale_mems * config.D)
    self.Xi = jr.normal(key, (config.D, nmems))

  def energy(self, g:jnp.ndarray):
    """Return the Hopfield Network's energy"""
    hid = jnp.einsum("nd,dm->nm", g, self.Xi) # nTokens, nMems
    E = -0.5 * (jax.nn.relu(hid) ** 2).sum()
    return E