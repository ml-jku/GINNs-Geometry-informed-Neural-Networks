import logging
import math
import datetime
import os.path
from pathlib import Path

import einops
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import wandb
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from PIL import Image
from io import BytesIO
from matplotlib.colors import LogNorm
import warnings

from models.point_wrapper import PointWrapper
from util.vis_utils import flip_arrow, generate_color, n_rows_cols
# Filter Warning - TODO: maybe fix this and other lines more properly later
# fig += k3d.mesh(verts, faces, group=group, color=MESH_COLOR, side='double', opacity=MESH_OPACITY, translation=[0, shape_grid[1]*i_col, shape_grid[2]*i_row]) 
warnings.filterwarnings("ignore", message="Given trait value dtype.*")

logging.getLogger('matplotlib').setLevel(logging.WARNING)
TRAJ_WIDTH = 0.005
## COLORS
MESH_COLOR = 0xe31b23
MESH_OPACITY = 1.0

# Points
DEFAULT_POINT_SIZE = 0.01
DEFAULT_POINT_OPACITY = 1.0
DEFAULT_POINT_SHADER = 'flat'
DEFAULT_PTS_COLOR = 0x808080
MINIMA_COLOR = 0x0000ff
MAXIMA_COLOR = 0xff0000
SADDLE_COLOR = 0x00ff00
# purple
ENDPOINT_COLOR = 0x800080

# Paths
DEFAULT_PATH_WIDTH = 0.005
DEFAULT_PATH_OPACITY = 1.0
PATH_SHADER = 'simple'
DEFAULT_PATH_COLOR = 0x606060
ASC_PATH_COLOR = 0xFFB3B3
DESC_PATH_COLOR = 0xB3B3FF


# STANDARD_COLORS
BLACK = 0x000000
GRAY = 0x808080
PASTEL_RED = 0xFFB3B3
PASTEL_GREEN = 0xB3FFB3
PASTEL_BLUE = 0xB3B3FF
PASTEL_YELLOW = 0xFFFFB3
PASTEL_PURPLE = 0xFFB3FF
PASTEL_CYAN = 0xB3FFFF
PASTEL_PEACH = 0xFFE5B4
PASTEL_LAVENDER = 0xE6B3FF
PASTEL_MINT = 0xB3FFC9
PASTEL_ROSE = 0xFFB3DF
PASTEL_SKY_BLUE = 0xB3E0FF
PASTEL_LEMON = 0xFFFFB3
PASTEL_CORAL = 0xFFC1B3
PASTEL_TEAL = 0xB3FFF2
PASTEL_VIOLET = 0xCAB3FF

#              self.mpm.plot(self.ph_plotter.plot_ph_diagram, 'plot_shape', arg_list=[self.p_sampler.constr_pts_dict], kwargs_dict={})
 

class PHPlotter():
    def __init__(self, iso_level, fig_size=(12, 12)):
        self.fig_size = fig_size
        self.infty = 2
        self.ISO = iso_level

    def _n_rows_cols(self, n_shapes_to_plot):
        ''' Returns the number of n_rows and columns. Favor more n_rows over n_cols to better show in wandb side-by-side '''
        n_cols = int(math.sqrt(n_shapes_to_plot))
        n_rows = int(math.ceil(n_shapes_to_plot / n_cols))
        return n_rows, n_cols

    def plot_ph_diagram(self, PH):
        
        n_rows, n_cols = n_rows_cols(len(PH))
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=self.fig_size, squeeze=False)

        for i in range(0, n_rows):
            for j in range(0, n_cols):
                if i * n_cols + j >= len(PH):
                    break
                
                ax = axs[i, j]
                ph = PH[i*n_cols + j]

                betti_numbers = np.unique(ph[:, 0])
                infmask = ph[:, 2] <= 1.e100
                ax.vlines(ph[~infmask, 1], ph[~infmask, 1], self.infty)

                for b_n in betti_numbers:
                    ph_n = ph[infmask, :]
                    ph_n = ph_n[ph_n[:, 0]==b_n]
                    ax.scatter(ph_n[:, 1], ph_n[:, 2], marker='.', label=f'Betti_{int(b_n)}')


                ## Visual guides
                ax.plot([-1,1], [-1,1], c='k')
                rect = Rectangle((-self.infty, self.ISO), self.infty+self.ISO, self.infty+self.ISO, linewidth=1, edgecolor='none', facecolor='gray', alpha=0.2)
                ax.add_patch(rect)
                ## Format
                ax.axis('scaled')
                ph_0 = ph[ph[:, 0]==0]
                ax.set_xlim([ph_0[:, 1].min()-0.2, ph_0[:, 1].max()+0.2])
                ax.set_ylim([ph_0[:, 1].min()-0.2, ph_0[:, 1].max()+0.2])
                ax.legend(loc='lower right')
                # ax.set_xscale('log')
        return self.handle_plot_show_matplotlib(fig, fig_label="PH Diagram")


    def handle_plot_show_matplotlib(self, fig, fig_label):
        """ Handle piping the plot to different outputs. """
        ## Let wandb handle the saving (this has a lot of whitespace)
        ## Remove the whitespace
        with BytesIO() as buf:
            fig.savefig(buf, transparent=False, facecolor='w', dpi=100)
            buf.seek(0)
            wandb_img = wandb.Image(Image.open(buf))
        
        plt.close(fig) ## close figures, otherwise they just stack up
        
        return fig_label, wandb_img
    

