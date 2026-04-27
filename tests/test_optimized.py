import numpy as np
import numba
import time
try:
    from matplotlib import pyplot
    HAS_PYPLOT = True
except ImportError:
    HAS_PYPLOT = False

plotEvery = 25

@numba.jit(nopython=True)
def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def main():
    # Space
    Nx = 400 
    Ny = 100

    tau = .53
    Nt = 100  # REDUCED FOR TESTING

    #lattice speeds and weights 
    NL = 9 # Number of Lattices
    
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # initial conditions
    F = np.ones((Ny, Nx, NL)) + .01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] = 2.3

    # Vectorized cylinder mask creation
    y_coords = np.arange(Ny)[:, np.newaxis]
    x_coords = np.arange(Nx)[np.newaxis, :]
    distances = np.sqrt((x_coords - Nx // 4) ** 2 + (y_coords - Ny // 2) ** 2)
    cylinder = distances < 13 

    # Pre-compute cylinder properties
    cylinder_indices = np.where(cylinder)
    
    # main loop
    t_start = time.perf_counter()
    for t in range(Nt):
        if t % 50 == 0:
            print(f"Iteration {t}/{Nt}")

        # Streaming with boundary conditions using manual roll
        for i in range(NL):
            F[:, :, i] = np.roll(F[:, :, i], cxs[i], axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cys[i], axis=0)
        
        # Edge boundary conditions
        F[:, -1, 6] = F[:, -2, 6]
        F[:, -1, 7] = F[:, -2, 7]
        F[:, -1, 8] = F[:, -2, 8]
        
        F[:, 0, 2] = F[:, 1, 2]
        F[:, 0, 3] = F[:, 1, 3]
        F[:, 0, 4] = F[:, 1, 4]
        
        # Cylinder bounce-back boundary condition
        for idx in range(len(cylinder_indices[0])):
            y, x = cylinder_indices[0][idx], cylinder_indices[1][idx]
            # Swap opposite directions
            F[y, x, 1], F[y, x, 5] = F[y, x, 5], F[y, x, 1]
            F[y, x, 2], F[y, x, 6] = F[y, x, 6], F[y, x, 2]
            F[y, x, 3], F[y, x, 7] = F[y, x, 7], F[y, x, 3]
            F[y, x, 4], F[y, x, 8] = F[y, x, 8], F[y, x, 4]
        
        # Compute macroscopic variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho
        
        # Apply boundary conditions to velocity
        ux[cylinder] = 0
        uy[cylinder] = 0
        
        # Collision step (JIT compiled)
        _collision_step(F, rho, ux, uy, cxs, cys, weights, tau)

        if (t % plotEvery == 0) and HAS_PYPLOT:
            dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
            dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
            curl = dfydx - dfxdy

            pyplot.imshow(curl, cmap='bwr')
            pyplot.pause(0.01)
            pyplot.cla()
    
    t_end = time.perf_counter()
    print(f"\nTotal runtime: {t_end - t_start:.2f} seconds")
    print(f"Average per iteration: {(t_end - t_start) / Nt * 1000:.2f} ms")


@numba.jit(nopython=True)
def _collision_step(F, rho, ux, uy, cxs, cys, weights, tau):
    """Collision step with equilibrium calculation (Numba JIT compiled)."""
    Ny, Nx, NL = F.shape
    
    for i in range(NL):
        cx, cy, w = cxs[i], cys[i], weights[i]
        
        for y in range(Ny):
            for x in range(Nx):
                cu = cx * ux[y, x] + cy * uy[y, x]
                Feq = rho[y, x] * w * (
                    1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux[y, x] * ux[y, x] + uy[y, x] * uy[y, x])
                )
                F[y, x, i] += -(1.0 / tau) * (F[y, x, i] - Feq)


if __name__ == '__main__':
    main()
