import colorsys
import math


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def generate_color(i):
    r = (i & 1) * 255       # Red when i is odd
    g = ((i >> 1) & 1) * 255 # Green when the second bit of i is 1
    b = ((i >> 2) & 1) * 255 # Blue when the third bit of i is 1
    return r + (g << 8) + (b << 16)


def n_rows_cols(n_shapes, flatten=False):
        ''' Returns the number of n_rows and columns. Favor more n_rows over n_colsto better show in wandb side-by-side '''
        if flatten:
            return n_shapes, 1
        n_cols = int(math.sqrt(n_shapes))
        n_rows = int(math.ceil(n_shapes / n_cols))
        return n_rows, n_cols


def flip_arrow(xy, dxy):
    return xy + dxy, -1 * dxy