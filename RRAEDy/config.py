"""
Module defining different latent space dynamics models for RRAEDy.

In every script, the user should have

import RRAEDy.config 

as the first line, this will set how all RRAEs classes behave.

In other words, after importing RRAEDy.config, all RRAEs models will use the
latent dynamics model defined here, you can just import RRAEs classes as usual,
and these will use the latent dynamics defined in this file.

The default latent dynamics are RR_DMD, which implements RRAEDy.

Other models include Neural_ODE and RNN, they're there for those who want 
to experiment with different latent dynamics.
"""
from RRAEs.AE_base import AE_base as rraes
from RRAEs.config import Autoencoder
import jax.numpy as jnp
import jax.random as jrandom
import jax
from RRAEs.utilities import stable_SVD, MLP_with_linear
import equinox as eqx
import jax.lax as lax

class RR_DMD(Autoencoder):
    def perform_in_latent(self, y, k_max=None, steps=None, apply_basis=None, get_W=False, apply_W=None, get_basis_coeffs=False, just_pass=False, ret_reg=False, ret_W=False, reg=False, RR=True, **kwargs):
        if RR:
            if apply_basis is None:
                def get_bases_coeffs(lat):
                    U, S, Vt = stable_SVD(lat)
                    coeffs = jnp.matmul(jnp.diag(S[:k_max]), Vt[:k_max, :])
                    basis = U[:, :k_max]
                    return basis, coeffs
                
                bases, alphas = get_bases_coeffs(y.reshape(y.shape[0], -1))
                alphas = alphas.reshape(k_max, *y.shape[1:])

            else:
                alphas = jax.vmap(lambda l: apply_basis.T @ l, in_axes=-2, out_axes=-2)(y)
                bases = apply_basis
        else:
            alphas = y
            bases = jnp.eye(y.shape[0])
        
        if get_basis_coeffs and (apply_W is None):
            return bases, alphas
        
        if just_pass:
            if RR:
                return jax.vmap(lambda c: bases @ c, in_axes=-1, out_axes=-1)(alphas), None
            else:
                return alphas, None


        if apply_W is None:
            alphat = alphas[:, :-1].reshape(alphas.shape[0], -1)
            alphatp = alphas[:, 1:].reshape(alphas.shape[0], -1)

            U, S, Vt = stable_SVD(alphat)

            W = alphatp @ Vt.T @ jnp.diag(1 / S) @ U.T

            if reg:
                s0 = stable_SVD(W)[1][0]
                reg_term = 1*(s0 < 1) + s0 * (s0 >= 1)
            else:
                reg_term = 1 
            W = W / reg_term
        else:
            W = apply_W

        if get_W:
            return W
        
        if steps is None:
            steps = alphas.shape[1]-1
        
        def to_map(i, a_vals):
            xs = jnp.arange(steps)
            def to_scan(aa, cc):
                coeff_next = W @ aa
                return coeff_next, coeff_next 

            alphanew = jax.lax.scan(to_scan, a_vals[..., 0:1], xs=xs)[1][..., 0].T
            alphanew = jnp.concatenate((a_vals[..., 0:1], alphanew), -1)
            if get_basis_coeffs:
                return bases, alphanew
            if RR:
                return bases @ alphanew
            else:
                return alphanew
            
        final_lat = jax.vmap(to_map, in_axes=-1, out_axes=-1)(jnp.arange(alphas.shape[-1]), alphas)
       
        if ret_reg:
          if ret_W:
              return final_lat, reg_term, W
          return final_lat, reg_term
        if ret_W:
            return final_lat, W
        return final_lat

class Neural_ODE(Autoencoder):
    neural_mlp: MLP_with_linear
    def __init__(self, *args, latent_ODE, **kwargs):
        super().__init__(*args, **kwargs)
        self.neural_mlp = MLP_with_linear(
            in_size=latent_ODE,
            out_size=latent_ODE,
            width_size=128,
            depth=3,
            key=jrandom.PRNGKey(42)
        )
    def perform_in_latent(self, y, *args, steps=None, dt=0.01, **kwargs):
        steps = y.shape[-2]-1 if steps is None else steps
        z0 = y[..., 0:1, :]

        def apply_f(z):
            lead_shape = z.shape[:-1]
            z_flat = z.reshape((-1, z.shape[-1]))
            f_flat = jax.vmap(self.neural_mlp)(z_flat)
            return f_flat.reshape(*lead_shape, z.shape[-1])

        def scan_step(carry, _):
            z_next = carry + dt * jax.vmap(jax.vmap(apply_f, in_axes=-1, out_axes=-1), in_axes=-1, out_axes=-1)(carry)
            return z_next, z_next

        xs = jnp.arange(steps)
        _, outs = jax.lax.scan(scan_step, z0, xs)
        outs = jnp.moveaxis(outs[:, :, 0], 0, -2)
        outs = jnp.concatenate((z0, outs), axis=-2)
        return outs

class Recurrent_Net(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear
    bias: jax.Array

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))
        def f(carry, inp):
            return self.cell(inp, carry), None
        out, _ = lax.scan(f, hidden, input)
        return self.linear(out) + self.bias
    
class RNN(Autoencoder):
    rnn: Recurrent_Net
    def __init__(self, *args, latent_ODE, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnn = Recurrent_Net(latent_ODE, latent_ODE, 128, key=jrandom.PRNGKey(0))

    def perform_in_latent(self, y, *args, steps=None, **kwargs):
        steps = y.shape[-2]-1 if steps is None else steps
        z0 = y[..., :, :]

        def scan_step(carry, _):
            z_next = jax.vmap(self.rnn, in_axes=-1, out_axes=-1)(carry)
            return z_next, z_next

        xs = jnp.arange(steps)
        _, outs = jax.lax.scan(scan_step, z0, xs)
        outs = jnp.moveaxis(outs[:, :, 0], 0, -2)
        outs = jnp.concatenate((y[..., 0:1, :], outs), axis=-2)
        return outs

rraes.set_autoencoder_base(RR_DMD)