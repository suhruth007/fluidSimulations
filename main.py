import numpy as np
from matplotlib import pyplot

plotEvery = 25

def distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def main():
    # Space
    Nx = 400 
    Ny = 100

    tau = .53
    Nt = 30000 # time 

    #lattice speeds and weights 
    NL = 9 # Number of Lattices
    
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    # initial conditions
    F = np.ones((Ny, Nx, NL)) + .01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] = 2.3

    cylinder = np.full((Ny, Nx), False)

    # define cylinder area
    for y in range(Ny):
        for x in range(Nx):
            if (distance(Nx //4 ,Ny // 2, x, y) < 13):
                cylinder[y][x] = True 

    # main loop
    for t in range(Nt):
        print(t)

        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]

        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis = 1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis = 0)

        bndryF = F[cylinder, :] 
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]] # inverting the condition when collision is hit

        #fluid variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho


        F[cylinder, :] = bndryF
        ux[cylinder] = 0
        uy[cylinder] = 0

        # Collision 
        Feq = np.zeros(F.shape)

        for i, cx, cy, w in zip(range(0, NL), cxs, cys, weights):
            Feq[:, :, i] = rho * w *(
                1 + 3 * (cx*ux + cy*uy) + 9 * (cx*ux + cy*uy) ** 2 / 2 -3 * (ux ** 2 + uy ** 2)/ 2
            )
        
        F += -(1/tau) * (F - Feq)



        if (t % plotEvery == 0):
            dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
            dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
            curl = dfydx - dfxdy

            pyplot.imshow(curl, cmap='bwr')
            pyplot.pause(0.1)
            pyplot.cla()



if __name__ == '__main__':
    main()