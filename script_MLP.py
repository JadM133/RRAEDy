import RRAEDy.config # always first
from RRAEs.AE_classes import RRAE_MLP
import jax.random as jrandom
from RRAEDy.training_classes import DMD_RRAE_Trainor_class
import jax.numpy as jnp
import equinox as eqx
from RRAEDy.utilities import get_data
from RRAEs.trackers import RRAE_gen_Tracker

if __name__ == "__main__":
    # Step 1: Get the data - replace this with your own data of the same shape.
    # The data (i.e. both x_train and y_train), should be of shape (D x T x N), 
    # where D is the number of features, T is the number of timesteps, and N 
    # is the number of samples.
    
    # y_train and y_test should be equal to x_train and x_test

    # Default values for other parameters:
    # p_train = p_test = None
    # pre_func_inp = pre_func_out = lambda x: x
    # kwargs = {}
    problem = "van_der_pol"

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

    # Step 2: Specify the method and model to use:
    method = "RRAE_DMD"
    RR = True
    just_pass = False

    model_cls = RRAE_MLP

    loss_type = "Strong"  # Specify the loss type, according to the model chosen.

    # Step 3: Specify the archietectures' parameters:
    latent_size = 10 # br64  # latent space dimension  # 10 for VDP
    k_max = 9 # number of features in the latent space (after the truncated SVD).

    # Step 4: Define your trainor, with the model, data, and parameters.
    trainor = DMD_RRAE_Trainor_class(
        x_train,
        model_cls,
        in_size=x_train.shape[0],
        latent_size=latent_size,
        k_max=k_max,
        kk=k_max,
        call_map_count=2, # vectorize over both time and samples for encoder/decoder
        folder=f"{problem}_init", # folder where to save
        file=f"{method}_{problem}_test.pkl", # file where to save
        norm_in="None", # can also use "minmax" or "meanstd"
        norm_out="None", # use the same as norm_out since we have an autoencoder
        out_train=x_train,
        key=jrandom.PRNGKey(5)
    )

    # Step 5: Define the loss function and training parameters:
    norm_loss_ = lambda x, y: jnp.linalg.norm(x-y)/jnp.linalg.norm(y)*100

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fun(diff_model, static_model, input, out, idx, *, k_max, reg=False, kwargs_model={},**kwargs):
        model = eqx.combine(diff_model, static_model)
        lat, reg, W = model.latent(input, k_max=k_max, inv_norm_out=False, ret_reg=True, ret_W=True, reg=reg, RR=RR, just_pass=just_pass, **kwargs_model)
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

    # Forecasting, give only initial conditions, provide both the 
    # basis and W (saved in trainor), and ask to predict as many 
    # steps as u need.

    pr = trainor.model(pre_func_inp(x_train[..., 0:1, -100:]), p=None, apply_W=trainor.W, apply_basis=trainor.basis, steps=x_train.shape[1]-1)
    out = pre_func_out(x_train[:, :, -100:])
    print(jnp.linalg.norm(pr-out)/jnp.linalg.norm(out)*100)

    pr = trainor.model(pre_func_inp(x_test[..., 0:1, :]), p=None, apply_W=trainor.W, apply_basis=trainor.basis, steps=x_test.shape[1]-1)
    out = pre_func_out(x_test)
    print(jnp.linalg.norm(pr-out)/jnp.linalg.norm(out)*100)
