"""Utility functions for RRAEDy."""
import jax.numpy as jnp
import equinox as eqx

def loss_generator(which=None, norm_loss_=None):
    """ Definied to be compatible with RRAEs. Only one loss structure for RRAEDy is implemented here.
    The loss can be modified by passing a different norm_loss_ function.
    """
    if norm_loss_ is None:
        norm_loss_ = lambda x1, x2: jnp.linalg.norm(x1 - x2) / jnp.linalg.norm(x2) * 100

    if which == "default":
        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(diff_model, static_model, input, out, idx, *, k_max, kwargs_model={},**kwargs):
            model = eqx.combine(diff_model, static_model)
            if "p" in kwargs:
                pred = model(input, p=kwargs["p"][..., idx], k_max=k_max, inv_norm_out=False, **kwargs_model)
            else:
                pred = model(input, k_max=k_max, inv_norm_out=False, **kwargs_model)
            aux = {"loss": norm_loss_(pred, out), "k_max": k_max}
            return norm_loss_(pred, out), (aux, {"reg": 1.0})
    else:
        raise NotImplementedError

    return loss_fun

def get_data(problem, folder=None):
    """Function that generates the examples presented in the paper."""
    match problem:
        case "circular_gaussian":
            import jax.numpy as jnp
            import jax.random as jrandom

            D = 64
            Ntr = 1000
            Nte = 100
            T = 200  # Number of time samples
            sigma = 0.1

            def gaussian_2d(x, y, x0, y0, sigma):
                return jnp.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

            x = jnp.linspace(-1, 1, D)
            y = jnp.linspace(-1, 1, D)
            X, Y = jnp.meshgrid(x, y)

            # Radii for training samples (e.g., different circular paths)
            radii_train = jnp.linspace(0.3, 0.8, Ntr)
            theta = jnp.linspace(0, 3 * jnp.pi, T)  # Full circle over T steps

            # Create training data
            train_data = []
            for i in range(Ntr):
                r = radii_train[i]
                traj = []
                for t in range(T):
                    x_t = r * jnp.cos(theta[t])
                    y_t = r * jnp.sin(theta[t])
                    traj.append(gaussian_2d(X, Y, x_t, y_t, sigma))
                traj = jnp.stack(traj, axis=-1)  # (D, D, T)
                train_data.append(traj)
            train_data = jnp.stack(train_data, axis=-2)  # (D, D, Ntr, T)

            # Test data: use random radii
            key = jrandom.PRNGKey(0)
            radii_test = jrandom.uniform(key, (Nte,), minval=0.3, maxval=0.8)

            test_data = []
            for i in range(Nte):
                r = radii_test[i]
                traj = []
                for t in range(T):
                    x_t = r * jnp.cos(theta[t])
                    y_t = r * jnp.sin(theta[t])
                    traj.append(gaussian_2d(X, Y, x_t, y_t, sigma))
                traj = jnp.stack(traj, axis=-1)  # (D, D, T)
                test_data.append(traj)
            test_data = jnp.stack(test_data, axis=-2)  # (D, D, Nte, T)

            mean = jnp.mean(train_data)
            std = jnp.std(train_data)
            train_data = (train_data - mean) / std
            test_data = (test_data - mean) / std
            print(mean, std)
            
            # Add channel dimension: (1, D, D, N, T)
            x_train = jnp.expand_dims(train_data, 0)
            x_test = jnp.expand_dims(test_data, 0)
            x_train = jnp.swapaxes(x_train, -1, -2)  # (1, D, D, T, N) → (1, D, D, N, T)
            x_test = jnp.swapaxes(x_test, -1, -2)
            y_train = x_train
            y_test = x_test

            # Shuffle training samples (use same permutation for data and parameters)
            key = jrandom.PRNGKey(0)
            perm = jrandom.permutation(key, x_train.shape[-1])
            x_train = jnp.take(x_train, perm, axis=-1)
            y_train = jnp.take(y_train, perm, axis=-1)
            radii_train = jnp.take(radii_train, perm, axis=0)

            # Parameters: just the radius used
            p_train = radii_train.reshape(-1, 1)
            p_test = radii_test.reshape(-1, 1)
            return x_train, x_test, p_train, p_test, y_train, y_test, lambda x: x, lambda x: x, ()

        case "fluid":
            import numpy as np
            import jax.numpy as jnp
            import jax.random as jrandom
            import os 

            try:
                data = np.load(os.path.join(folder, "cylinder_vortex_data_original.npy"))
                data = data.reshape((1, 100, 50, data.shape[1], data.shape[2]), order="F")
                ps = np.load(os.path.join(folder, "inlet.npy"))
            except:
                raise FileNotFoundError(f"Data files not found in folder: {folder}, run fluid_data_gen.py to generate data first.")
            
            N = data.shape[-1]
            indices = jnp.arange(N)
            key = jrandom.PRNGKey(0)
            shuffled_indices = jrandom.permutation(key, indices)
            shuffled_indices = np.array(shuffled_indices, dtype=int)
            split = int(N * 0.85)
            train_idx = shuffled_indices[:split]
            test_idx = shuffled_indices[split:]

            _x_train = data[..., train_idx]
            _x_test = data[..., test_idx]
            _y_train = _x_train
            _y_test = _x_test
            p_train = ps[train_idx]
            p_test = ps[test_idx]
            pre_func_inp = lambda x: x
            pre_func_out = lambda x: x

            _x_train = _x_train[:, ::5]
            _y_train = _y_train[:, ::5]
            _x_test = _x_test[:, ::5]
            _y_test = _y_test[:, ::5]
            
            return _x_train, _x_test, p_train, p_test, _y_train, _y_test, pre_func_inp, pre_func_out, (shuffled_indices,)

    
        case "burgers":

            import pickle
            pre_func_out = lambda x:x
            pre_func_inp = lambda x:x
            try:
                with open(f"{folder}/burgers_data.pkl", "rb") as f:
                    _x_train, _x_test, p_train, p_test, _y_train, _y_test = pickle.load(f)
                _x_train = _x_train.T
                _y_train = _y_train.T
                _x_test = _x_test.T
                _y_test = _y_test.T

                return _x_train, _x_test, p_train, p_test, _y_train, _y_test, pre_func_inp, pre_func_out, ()
            except FileNotFoundError:
                raise FileNotFoundError(f"Data file not found in folder: {folder}, run burgers_data_gen.py to generate data first.")

        case "mass_spring_param":
            import numpy as np
            import jax.numpy as jnp
            import jax.random as jrandom

            N = 2000
            T = 1000
            t = np.linspace(0, 15, T)

            x0 = 1.0
            v0 = 0.0

            m_vals = np.random.uniform(0.5, 2.0, N)
            c_vals = np.random.uniform(0.1, 2.0, N)
            k_vals = np.random.uniform(0.5, 3.0, N)

            x = np.zeros((T, N))
            v = np.zeros((T, N))

            for i in range(N):
                m = m_vals[i]
                c = c_vals[i]
                k = k_vals[i]

                discriminant = c**2 - 4 * m * k

                if discriminant > 0:
                    # Overdamped
                    r1 = (-c + np.sqrt(discriminant)) / (2 * m)
                    r2 = (-c - np.sqrt(discriminant)) / (2 * m)
                    A = (v0 - r2 * x0) / (r1 - r2)
                    B = x0 - A
                    x[:, i] = A * np.exp(r1 * t) + B * np.exp(r2 * t)
                    v[:, i] = A * r1 * np.exp(r1 * t) + B * r2 * np.exp(r2 * t)
                elif discriminant == 0:
                    # Critically damped
                    r = -c / (2 * m)
                    A = x0
                    B = v0 - r * x0
                    x[:, i] = (A + B * t) * np.exp(r * t)
                    v[:, i] = (B + r * (A + B * t)) * np.exp(r * t)
                else:
                    # Underdamped
                    real_part = -c / (2 * m)
                    imag_part = np.sqrt(-discriminant) / (2 * m)
                    A = x0
                    B = (v0 - real_part * x0) / imag_part
                    x[:, i] = np.exp(real_part * t) * (A * np.cos(imag_part * t) + B * np.sin(imag_part * t))
                    v[:, i] = np.exp(real_part * t) * (
                        -A * imag_part * np.sin(imag_part * t)
                        + B * imag_part * np.cos(imag_part * t)
                        + real_part * (A * np.cos(imag_part * t) + B * np.sin(imag_part * t))
                    )

            input = jnp.stack((x.T, v.T), -1)
            params = jnp.stack((m_vals, c_vals, k_vals), axis=-1)

            # Split into train/test
            split_idx = 200
            train_input = input[:split_idx].T 
            test_input = input[split_idx:].T
            train_params = params[:split_idx]
            test_params = params[split_idx:]

            return train_input, test_input, train_params.T, test_params.T, train_input, test_input, lambda x: x, lambda x: x, ()
            

        case "van_der_pol":
          import numpy as np
          import jax.numpy as jnp
          from scipy.integrate import solve_ivp
          import jax.random as jrandom

          mu = 2.0

          t_max = 10.0
          T = 1000
          t = np.linspace(0, t_max, T)

          N = 2000  # number of trajectories

          x0s = jrandom.uniform(jrandom.key(50), minval=-1.5, maxval=1.5, shape=(N,))
          v0s = jrandom.uniform(jrandom.key(137), minval=-1.5, maxval=1.5, shape=(N,))

          x = np.zeros((T, N))
          v = np.zeros((T, N))

          def van_der_pol(t, y):
              x, dx = y
              ddx = mu * (1 - x**2) * dx - x
              return [dx, ddx]

          for i in range(N):
              y0 = [x0s[i], v0s[i]]
              sol = solve_ivp(van_der_pol, [0, t_max], y0, t_eval=t, method='RK45', rtol=1e-6, atol=1e-9)
              x[:, i] = sol.y[0]
              v[:, i] = sol.y[1]

          input = jnp.stack((x.T, v.T), axis=0)

          # Split into train/test
          split_idx = int(0.8 * N)
          train_input = input[:, :split_idx].transpose(0, 2, 1)
          test_input = input[:, split_idx:].transpose(0, 2, 1)
        
          _x_train = train_input[:, ::5]
          _y_train = train_input[:, ::5]
          _x_test = test_input[:, ::5]
          _y_test = test_input[:, ::5]
          
          return train_input, test_input, None, None, train_input, test_input, lambda x: x, lambda x: x, ()
        case _:
            raise NotImplementedError(f"Problem {problem} not implemented in data generator.")