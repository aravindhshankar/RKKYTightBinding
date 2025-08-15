## specification of unit cell distances 
rvals = np.array([0, 1, 2, 3, 4, 10, 11, 12, 50, 51, 52, 99, 100, 101])

## specification of which sublattice elements they connect:
return np.array((Gkx[0, 0], Gkx[1, 1], Gkx[1, 2], Gkx[2, 1]))
    0: site A - A 
    1: site B - B
    2: site A - B
    3: site B - A
### I suppose we just need sites 0 and 2 for the 2imp nrg: same sublattice and different sublattice

