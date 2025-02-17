import numpy as np

def tensor_product_xz(x, z):
    """
    Generate correct inputs for bx different xs and bz different zs.
    For each z we want to evaluate it at all xs and vice versa, so we need a tensor product between the rows of the two vectors.
    x: [bx, nx]
    z: [bz, nz]
    returns: ([bx*bz, nx], [bz*bx, nz])
    e.g.:
    x = [[1, 2], [3, 4]]
    z = [[5, 6], [7, 8]]
    x_tp = [[1, 2], [3, 4], [1, 2], [3, 4]]
    z_tp = [[5, 6], [5, 6], [7, 8], [7, 8]]
    """
    z_tp = z.repeat_interleave(repeats=len(x), dim=0)
    x_tp = x.repeat(len(z), 1)
    return x_tp, z_tp


def tensor_product_xz_np(x, z):
    """
    Generate correct inputs for bx different xs and bz different zs.
    For each z we want to evaluate it at all xs and vice versa, so we need a tensor product between the rows of the two vectors.
    x: [bx, nx]
    z: [bz, nz]
    returns: ([bx*bz, nx], [bz*bx, nz])
    e.g.:
    x = [[1, 2], [3, 4]]
    z = [[5, 6], [7, 8]]
    x_tp = [[1, 2], [3, 4], [1, 2], [3, 4]]
    z_tp = [[5, 6], [5, 6], [7, 8], [7, 8]]
    """
    bx, nx = x.shape
    bz, nz = z.shape

    # Repeat each row of z for bx times
    z_tp = np.repeat(z, bx, axis=0)

    # Tile x for bz times
    x_tp = np.tile(x, (bz, 1))

    return x_tp, z_tp