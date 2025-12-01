""" Script to generate Burgers' equation simulation data with quartic polynomial initial conditions. """
import jax.random as jrandom
import numpy as np
from scipy.integrate import solve_ivp
import pickle

def quartic_poly(x, a, b, m0, m1, alpha):
    d1 = b - a - m0
    d2 = m1 - m0
    
    c0 = a
    c1 = m0
    c4 = alpha
    c3 = d2 - 2*d1 - 2*alpha
    c2 = 3*d1 - d2 + alpha
    
    return c4*x**4 + c3*x**3 + c2*x**2 + c1*x + c0

def generate_quartic_samples(key, N, a=0, b=0,
                            m0_range=(-3,3),
                            m1_range=(-3,3),
                            alpha_range=(-3,3),
                            x=None):
    if x is None:
        x = np.linspace(0,1,100)
    
    key, subkey1, subkey2, subkey3 = jrandom.split(key, 4)
    m0s = jrandom.uniform(subkey1, (N,), minval=m0_range[0], maxval=m0_range[1])
    m1s = jrandom.uniform(subkey2, (N,), minval=m1_range[0], maxval=m1_range[1])
    alphas = jrandom.uniform(subkey3, (N,), minval=alpha_range[0], maxval=alpha_range[1])
    
    def single_sample(i):
        return quartic_poly(x, a, b, m0s[i], m1s[i], alphas[i])
    
    samples = np.vstack([single_sample(i) for i in range(N)])
    
    params = {'m0': m0s, 'm1': m1s, 'alpha': alphas}
    return samples, params


if __name__ == "__main__":
    nu = 0.01  # viscosity
    L = 1.0    # spatial domain length
    Nx = 100   # spatial discretization points
    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]

    t = np.linspace(0, 15, 400)
    T = len(t)

    N = 2000 # Number of initial conditions

    key = jrandom.key(0)
    u0s = []

    def burgers_rhs(t, u):
        dudx = np.zeros_like(u)
        d2udx2 = np.zeros_like(u)
        dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
        dudx[0] = (u[1] - u[0]) / (dx)
        dudx[-1] = (u[-1] - u[-2]) / (dx)
        d2udx2[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / (dx**2)
        d2udx2[0] = d2udx2[1]
        d2udx2[-1] = d2udx2[-2]
        rhs = -u * dudx + nu * d2udx2
        rhs[0] = 0.0
        rhs[-1] = 0.0
        return rhs

    import jax.numpy as jnp

    def is_valid_curve(sol, threshold=0.05):
        # Check if the solution is too small, we don't want trivial cases
        interior = sol[1:-1]
        pos_max = jnp.max(interior)
        neg_min = jnp.min(interior)
        if (pos_max > 0) and (neg_min >= 0):
            return pos_max > threshold
        elif (neg_min < 0) and (pos_max <= 0):
            return jnp.abs(neg_min) > threshold
        elif (pos_max > 0) and (neg_min < 0):
            return (pos_max > threshold) and (jnp.abs(neg_min) > threshold)
        else:
            return False

    U = np.zeros((T, N, Nx))
    count = 0
    i = 0
    while i < N:
        key_i = jrandom.fold_in(key, count)
        u0, _ = generate_quartic_samples(key_i, 1)
        if is_valid_curve(u0[0]):
            sol = solve_ivp(burgers_rhs, [t[0], t[-1]], u0[0], t_eval=t, max_step=0.001, method='BDF', rtol=1e-6, atol=1e-8)
            print(sol.message)
            U[:, i, :] = sol.y.T
            i += 1
        count += 1

    # Train-test split
    split_idx = int(0.8 * N)
    train_data = U[:, :split_idx, :]
    test_data = U[:, split_idx:, :]

    train_data = np.transpose(train_data, (1, 0, 2))
    test_data = np.transpose(test_data, (1, 0, 2))

    with open("burgers_data.pkl", "wb") as f:
        pickle.dump((train_data, test_data, None, None, train_data, test_data, lambda x: x, lambda x: x, ()), f)
