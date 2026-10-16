import jax.numpy as jnp


def postprocessing(input, Pl, D):
    scale = jnp.exp(input[1]) * 1e-10 * D**2
    n = Pl.shape[-1] // 6
    return jnp.concatenate((Pl[:3*n]*scale, Pl[3*n:]*scale**3),axis=-1)