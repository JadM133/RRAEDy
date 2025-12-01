""" Script to generate fluid dynamics simulation data around a cylinder. """
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Grid parameters
nx, ny = 100, 50
Lx, Ly = 2.2, 1.0
dx, dy = Lx/(nx-1), Ly/(ny-1)

# Time parameters
nt = 1000
dt = 0.0005

# Fluid properties
nu = 0.01
rho = 1.0

cx, cy, r = 0.2, 0.5, 0.1 # Cylinder parameters

N = 200 # Number of simulations
inlet_velocities = np.linspace(1, 7.0, N)
T = nt
F = nx * ny

data = np.zeros((F, T, N))

x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

def inside_cylinder(x, y):
    return (x - cx)**2 + (y - cy)**2 <= r**2

mask = inside_cylinder(X, Y)

def build_poisson(nx, ny, dx, dy):
    N = nx * ny
    A = lil_matrix((N, N))
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            if i==0 or i==nx-1 or j==0 or j==ny-1:
                A[idx, idx] = 1.0
            else:
                A[idx, idx] = -2.0*(1.0/dx**2 + 1.0/dy**2)
                A[idx, idx-1] = 1.0/dx**2
                A[idx, idx+1] = 1.0/dx**2
                A[idx, idx-nx] = 1.0/dy**2
                A[idx, idx+nx] = 1.0/dy**2
    return csr_matrix(A)

A_p = build_poisson(nx, ny, dx, dy)

import pdb; pdb.set_trace()

for sim_i, u_inlet in enumerate(inlet_velocities):
    print(f"\nSimulation {sim_i+1}/{N} — u_inlet = {u_inlet}")

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    u_prev = np.zeros_like(u)
    v_prev = np.zeros_like(v) 
    p = np.zeros((ny, nx))

    for n in range(nt):
        u_prev[:,:] = u
        v_prev[:,:] = v

        # Boundary conditions
        u[:,0] = u_inlet
        v[:,0] = 0.0
        u[:,-1] = u[:,-2]
        v[:,-1] = v[:,-2]
        u[0,:] = 0.0
        u[-1,:] = 0.0
        v[0,:] = 0.0
        v[-1,:] = 0.0
        u[mask] = 0.0
        v[mask] = 0.0

        u_star = np.copy(u)
        v_star = np.copy(v)

        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if mask[j,i]: continue
                u_adv = u_prev[j,i]*(u_prev[j,i+1]-u_prev[j,i-1])/(2*dx) + \
                        v_prev[j,i]*(u_prev[j+1,i]-u_prev[j-1,i])/(2*dy)
                v_adv = u_prev[j,i]*(v_prev[j,i+1]-v_prev[j,i-1])/(2*dx) + \
                        v_prev[j,i]*(v_prev[j+1,i]-v_prev[j-1,i])/(2*dy)

                u_diff = nu * ( (u_prev[j,i+1]-2*u_prev[j,i]+u_prev[j,i-1])/(dx*dx) +
                                (u_prev[j+1,i]-2*u_prev[j,i]+u_prev[j-1,i])/(dy*dy) )
                v_diff = nu * ( (v_prev[j,i+1]-2*v_prev[j,i]+v_prev[j,i-1])/(dx*dx) +
                                (v_prev[j+1,i]-2*v_prev[j,i]+v_prev[j-1,i])/(dy*dy) )

                u_star[j,i] = u_prev[j,i] + dt * (-u_adv + u_diff)
                v_star[j,i] = v_prev[j,i] + dt * (-v_adv + v_diff)

        b = np.zeros((ny, nx))
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if mask[j,i]: continue
                div = ((u_star[j,i+1]-u_star[j,i-1])/(2*dx) +
                       (v_star[j+1,i]-v_star[j-1,i])/(2*dy))
                b[j,i] = (rho/dt) * div

        b_flat = b.flatten()
        p_flat = spsolve(A_p, b_flat)
        p = p_flat.reshape((ny, nx))

        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if mask[j,i]: continue
                u[j,i] = u_star[j,i] - dt*(p[j,i+1]-p[j,i-1])/(2*dx*rho)
                v[j,i] = v_star[j,i] - dt*(p[j+1,i]-p[j-1,i])/(2*dy*rho)

        u[mask] = 0.0
        v[mask] = 0.0

        # Computing vorticity
        vort = np.zeros_like(u)
        vort[1:-1,1:-1] = (v[1:-1,2:] - v[1:-1,0:-2])/(2*dx) - (u[2:,1:-1] - u[0:-2,1:-1])/(2*dy)
        data[:, n, sim_i] = vort.flatten()

        if n % 100 == 0:
            print(f"  Step {n}/{nt}")

np.save("cylinder_vortex_data.npy", data)
np.save("inlet.npy", np.expand_dims(inlet_velocities, 0))
print("\nData saved to 'cylinder_vortex_data.npy' and 'inlet.npy'")
