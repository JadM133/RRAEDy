from RRAEs.training_classes import RRAE_Trainor_class
from RRAEs.utilities import stable_SVD, eval_with_batches
import jax.numpy as jnp

class DMD_RRAE_Trainor_class(RRAE_Trainor_class):
    """ Trainor class for DMD-RRAE models. Inherits from RRAE_Trainor_class.
    Only adds the computation of the DMD matrix W after training.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_W(self, inp):
        
        def call_func(x):
            if isinstance(x, list):
                return self.model.latent(
                            self.pre_func_inp(x[0]), *x[1:], apply_basis=self.basis, get_basis_coeffs=True
                        )[1]
            else:
                return self.model.latent(
                            self.pre_func_inp(x), apply_basis=self.basis, get_basis_coeffs=True
                        )[1]
            
        alphas = eval_with_batches(
                inp,
                self.batch_size,
                call_func=call_func,
                str="Finding train latent space...",
                end_type="concat",
                key_idx=0,
            )

        alphat = alphas[:, :-1].reshape(alphas.shape[0], -1)

        alphatp = alphas[:, 1:].reshape(alphas.shape[0], -1)

        U, S, Vt = stable_SVD(alphat)

        W = alphatp @ Vt.T @ jnp.diag(1 / S) @ U.T
        
        self.W = W
        return self.W

    def fit(self, *args, **kwargs):
        
        if kwargs["ft_kwargs"].get("basis_input"):
            inp = kwargs["ft_kwargs"]["basis_input"]
        else:
            inp = args[0] if len(args) > 0 else kwargs["input"]

        super().fit(*args, **kwargs)
        self.get_W(inp)
