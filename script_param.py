import RRAEDy.config
from RRAEs.AE_classes import RRAE_MLP
import jax.random as jrandom
from RRAEDy.training_classes import DMD_RRAE_Trainor_class
import jax.numpy as jnp
import equinox as eqx
from RRAEDy.utilities import get_data
from RRAEs.trackers import RRAE_gen_Tracker
from RRAEs.utilities import MLP_with_linear
from RRAEs.wrappers import vmap_wrap
import jax

# Define a new RRAE_MLP class that takes parameters into account
# This is an example of how to extend the existing RRAEDy model
# Once this class is defined, the main is very similar to script_MLP.py
class RRAE_MLP_with_param(RRAE_MLP):
    _encode_mlp: MLP_with_linear
    _encode_mix: MLP_with_linear

    def __init__(self, *args, param_in, param_out, lat_mix, **kwargs):
        mlp_cls = vmap_wrap(MLP_with_linear, -1, 1)
        self._encode_mlp = mlp_cls(
                    in_size=param_in,
                    out_size=param_out,
                    width_size=64,
                    depth=1,
                    key=jrandom.key(0),
                )
        
        mlp_cls = vmap_wrap(MLP_with_linear, -1, 2)
        self._encode_mix = mlp_cls(
                    in_size=lat_mix,
                    out_size=kwargs["latent_size"],
                    width_size=64,
                    depth=1,
                    key=jrandom.key(0),
                )
        super().__init__(*args, **kwargs)
    
    def encode(self, x, p, *args, **kwargs):
        encoded_x = jax.vmap(jax.vmap(super().encode, in_axes=-1, out_axes=-1), in_axes=-1, out_axes=-1)(x, *args, **kwargs)
        encoded_p = jnp.repeat(self._encode_mlp(p)[:, None], encoded_x.shape[1], 1)
        return self._encode_mix(jnp.concatenate((encoded_x, encoded_p)))

    def decode(self, x):
        return jax.vmap(jax.vmap(super().decode, in_axes=-1, out_axes=-1), in_axes=-1, out_axes=-1)(x)
    
    def __call__(self, x, p, *args, **kwargs):
        return self.decode(super().perform_in_latent(self.encode(x, p), *args, **kwargs))

    def latent(self, x, p, *args, **kwargs):
        return super().perform_in_latent(self.encode(x, p), *args, **kwargs)
    
if __name__ == "__main__":
    problem = "mass_spring_param"

    (
        x_train,
        x_test,
        p_train,
        p_test,
        y_train,
        y_test,
        pre_func_inp,
        pre_func_out,
        kwargs,
    ) = get_data(problem)

    method = "RRAE_DMD"
    RR = True
    just_pass = False

    model_cls = RRAE_MLP_with_param

    loss_type = "Strong"  # Specify the loss type, according to the model chosen.

    latent_size = 10 # latent space dimension
    latent_size_p = 10 # latent space dimension for the parameters

    k_max = 9 # max number of features in the latent space (adaptive algorithm will reduce it)

    trainor = DMD_RRAE_Trainor_class(
        x_train,
        model_cls,
        in_size=x_train.shape[0],
        latent_size=latent_size,
        param_in=p_train.shape[0],
        param_out=latent_size_p,
        lat_mix=latent_size_p+latent_size,
        k_max=k_max,
        kwargs_enc={"depth": 3},
        kk=k_max,
        call_map_count=2,
        folder=f"{problem}_init", # folder where to save
        file=f"{method}_{problem}_test.pkl", # file where to save
        norm_in="None", # can also use "minmax" or "meanstd"
        norm_out="None", # use the same as norm_out
        out_train=x_train,
        nomap=True,
        key=jrandom.PRNGKey(5)
    )

    norm_loss_ = lambda x, y: jnp.linalg.norm(x-y)/jnp.linalg.norm(y)*100

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fun(diff_model, static_model, input, out, idx, *, k_max, reg=False, kwargs_model={},**kwargs):
        model = eqx.combine(diff_model, static_model)
        p_b = p_train[:, idx]
        # Only small modification below, to give p as input
        lat, reg, W = model.latent(input, p=p_b, k_max=k_max, inv_norm_out=False, ret_reg=True, ret_W=True, reg=reg, RR=RR, just_pass=just_pass, **kwargs_model)
        pred = model.decode(lat)
        aux = {"loss": norm_loss_(pred, out), "k_max": k_max, "W": W, "reg": reg}
        return norm_loss_(pred, out), (aux, {"reg": reg})

    training_kwargs = {
        "step_st": [150000],  # steps (forward/backward path for a batch)
        "batch_size_st": [64, 64], # batch size, reduce it if u have memory issues, but it has to be > k_max
        "lr_st": [1e-3, 1e-5, 1e-6, 1e-7, 1e-8],  # learning rate
        "print_every": 1,
        "loss_type": loss_fun,
        "latent_size": latent_size,
        "flush": True,
        "save_losses": True,
        # below we specify a gen tracker (k_max changes during training) 
        "tracker": RRAE_gen_Tracker(k_max, perf_loss=3, patience_init=4000, patience_not_right=5000, k_min=5, converged_steps=3000)
        # if you want to train with a fixed k_max, use the line below instead:
        # "tracker": RRAE_fixed_Tracker(k_max)
    }

    ft_end_type = "first" if just_pass else "concat"

    ft_kwargs = {
        "step_st": [0], # we usually don't need fine tuning, so set to 0
        "batch_size_st": [20],
        "lr_st": [1e-4, 1e-6, 1e-7, 1e-8],
        "print_every": 1,
        "latent_size": latent_size,
        "flush": True,
        "ft_end_type": ft_end_type,
        "basis_call_kwargs": {"RR": RR},
        "basis_input": [x_train, p_train]
    }

    trainor.fit(
        x_train[:, :],
        y_train[:, :],
        training_key=jrandom.key(50),
        training_kwargs=training_kwargs,
        ft_kwargs=ft_kwargs,
        pre_func_inp=pre_func_inp,
        pre_func_out=pre_func_out,
    )

    trainor.save_model()

    # Same forecasting as in script_MLP, but now with parameters
    pr = trainor.model(pre_func_inp(x_train[..., 0:1, -100:]), p=p_train[:, -100:], apply_W=trainor.W, apply_basis=trainor.basis, steps=x_train.shape[1]-1)
    out = pre_func_out(x_train[:, :, -100:])
    print(jnp.linalg.norm(pr-out)/jnp.linalg.norm(out)*100)


    pr = trainor.model(pre_func_inp(x_test[..., 0:1, :]), p=p_test, apply_W=trainor.W, apply_basis=trainor.basis, steps=x_test.shape[1]-1)
    out = pre_func_out(x_test)
    print(jnp.linalg.norm(pr-out)/jnp.linalg.norm(out)*100)
