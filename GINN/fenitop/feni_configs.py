from functools import partial
import time
import numpy as np
from dolfinx.mesh import create_rectangle, create_box, CellType
from mpi4py import MPI

from GINN.fenitop.simjeb_bc import SimjebBC

# a debugging helper function
def fn_print_matches(x, fn, name):
    res = fn(x)
    print(f'{name} matches: {res.sum()}')
    return res 

def get_configs(problem_str, bounds, resolution, envelope_mesh=None, interface_pts=None, F_jeb=None, **kwargs):
    
    if problem_str in ['mbb_beam_2d', 'cantilever_2d']:
        mesh_serial = create_rectangle(MPI.COMM_SELF, [bounds[:,0], bounds[:,1]],
                                resolution, CellType.quadrilateral)
    elif problem_str in ['simjeb']:
        mesh_serial = create_box(MPI.COMM_SELF, [bounds[:,0], bounds[:,1]],
                                resolution, CellType.hexahedron)
    
    fem = FEM_DICT[problem_str]
    fem["mesh"] = mesh_serial
    fem["mesh_serial"] = mesh_serial
    opt = OPT_DICT[problem_str]
    
    if problem_str in ['mbb_beam_2d', 'cantilever_2d']:
        return fem, opt
    
    if problem_str == 'simjeb':
        # define important FEM and OPT parameters for simjeb
            assert envelope_mesh is not None, "Envelope mesh must be provided for simjeb problem"
            assert interface_pts is not None, "Interface points must be provided for simjeb problem"
            assert F_jeb is not None, "JEB force vector must be provided for simjeb problem"
            
            jeb_bc = SimjebBC(interface_pts, bounds)
            
            # tuned visually; we make the bottom BC radii bigger, as the bottom interfaces have holes in the boundary plane
            proximity_top = 0.08
            proximity_bottom = 0.18
            
            # mechanism to easily choose loadcase in config
            F_jeb_dict = F_jeb[F_jeb['load_case']]
            
            traction_bcs = []
            # forces in x-y direction get boundary points at the back wall
            # force at the top left interface
            traction_bcs.append([tuple(F_jeb_dict['back_left']), lambda x: jeb_bc.top_left_interface_back_bc_mask(x, proximity_top)])
            # force at the top right interface
            traction_bcs.append([tuple(F_jeb_dict['back_right']), lambda x: jeb_bc.top_right_interface_back_bc_mask(x, proximity_top)])
            
            # forces in z-direction get boundary points at the top wall
            traction_bcs.append([tuple(F_jeb_dict['top_left']), lambda x: jeb_bc.top_left_interface_top_bc_mask(x, proximity_top)])
            traction_bcs.append([tuple(F_jeb_dict['top_right']), lambda x: jeb_bc.top_right_interface_top_bc_mask(x, proximity_top)])
            
            fem["traction_bcs"] = traction_bcs
            
            print(f'INFO: using all 4 bottom interfaces for BCs')
            fem["disp_bc"] = lambda x: jeb_bc.bottom_left_front_interface_bc_mask(x, proximity_bottom) | jeb_bc.bottom_right_front_interface_bc_mask(x, proximity_bottom) \
                | jeb_bc.bottom_left_back_interface_bc_mask(x, proximity_bottom) | jeb_bc.bottom_right_back_interface_bc_mask(x, proximity_bottom)
        
            
            # TODO: this is is useful for setting void/solid zones
            # check: this might be extremely slow! One could also do a point-comparison
            def envelope_mesh_contains(x):
                start_t = time.time()
                contains = envelope_mesh.contains(x.T)
                print(f'envelope mesh contains time: {time.time() - start_t}')
                print(f'x.shape {x.shape}, contains.sum() {contains.sum()}')
                return contains
            opt['void_zone'] = partial(fn_print_matches, name='void_zone', fn=lambda x: ~ envelope_mesh_contains(x))
            # points should be inside envelope close to interface
            def close_to_if_points(x):
                # r_bot = 0.08
                # bottom_mask = jeb_bc.bottom_left_front_interface_bc_mask(x, r_bot) | jeb_bc.bottom_right_front_interface_bc_mask(x, r_bot) \
                #     | jeb_bc.bottom_left_back_interface_bc_mask(x, r_bot) | jeb_bc.bottom_right_back_interface_bc_mask(x, r_bot)
                    
                # r_top = 0.16
                # top_mask = jeb_bc.top_left_interface_top_bc_mask(x, r_top) | jeb_bc.top_right_interface_top_bc_mask(x, r_top)
                dists = np.linalg.norm(x.T[:,None] - interface_pts[None], axis=-1)
                min_dist = np.min(dists, axis=1)
                # mask = min_dist < 0.08
                mask = min_dist < 0.08
                print(f'close to interface points: {mask.sum()}')
                return mask 
            opt['solid_zone'] = partial(fn_print_matches, name='solid_zone', 
                                        fn=lambda x: close_to_if_points(x) & envelope_mesh_contains(x))
            return fem, opt
        
    raise ValueError(f"Problem string {problem_str} not recognized")


FEM_DICT = {
    'mbb_beam_2d': {  # FEM parameters
            # "mesh": mesh_serial,
            # "mesh_serial": mesh_serial,
            "young's modulus": 196.0,
            "poisson's ratio": 0.30,
            "disp_bc1": lambda x: (np.isclose(x[0], 0.0)),
            "disp_bc2": lambda x: ((np.greater_equal(x[0], 2.9)) & (np.isclose(x[1], 0.0))),
            "traction_bcs": [
                    [[0, -10.0], # the traction force vector
                    lambda x: (np.less(x[0], 0.1) & np.isclose(x[1], 1.))]
                ],
            "body_force": (0, 0),
            "quadrature_degree": 2,
            "petsc_options": {
                "ksp_type": "cg",
                "pc_type": "gamg",
            },
        },
    'cantilever_2d': {  # FEM parameters
            # "mesh": mesh_serial,
            # "mesh_serial": mesh_serial,
            "young's modulus": 196.0,
            "poisson's ratio": 0.30,
            "disp_bc": lambda x: np.isclose(x[0], 0.),
            "traction_bcs": [
                    [[0, -1.0], # the traction force vector
                    lambda x: (np.isclose(x[0], 1.5) & np.greater(x[1], 0.1) & np.less(x[1], 0.2)) | 
                              (np.isclose(x[0], 1.5) & np.greater(x[1], 0.8) & np.less(x[1], 0.9))]
                ],
            "body_force": (0, 0),
            "quadrature_degree": 2,
            "petsc_options": {
                "ksp_type": "cg",
                "pc_type": "gamg",
            },
        },

    
    'simjeb': {  # FEA parameters
            # "mesh": mesh,
            # "mesh_serial": mesh_serial,
            "young's modulus": 100,
            "poisson's ratio": 0.25,
            # "disp_bc": lambda x: np.isclose(x[1], 0) & (np.less(x[0], 1.5) | np.greater(x[0], 8.5)),
            # "traction_bcs": [[(0, 0, -2.0),
            #                 lambda x: np.isclose(x[1], 30) & (
            #                     np.greater(x[0], 4.5) & np.less(x[0], 5.5)
            #                     & np.greater(x[2], 4.5) & np.less(x[2], 5.5))]],
            "body_force": (0, 0, 0),
            "quadrature_degree": 2,
            "petsc_options": {
                "ksp_type": "cg",
                "pc_type": "gamg",
            },
        },
}

OPT_DICT = {
    'mbb_beam_2d': {  # Topology optimization parameters
            "max_iter": 100,
            "opt_tol": 1e-5,
            # "vol_frac": 0.5,
            "solid_zone": lambda x: np.full(x.shape[1], False),
            "void_zone": lambda x: np.full(x.shape[1], False),
            # "penalty": 3.0,
            "epsilon": 1e-6,
            # "filter_radius": 1.2,  # for domain size (60, 20)
            "beta_interval": 50,
            "beta_max": 128,
            "use_oc": True,
            "move": 0.02,
            "opt_compliance": True,
        },
    'cantilever_2d': {  # Topology optimization parameters
            "max_iter": 100,
            "opt_tol": 1e-5,
            # "vol_frac": 0.5,
            "solid_zone": lambda x: np.full(x.shape[1], False),
            "void_zone": lambda x: np.full(x.shape[1], False),
            # "penalty": 3.0,
            "epsilon": 1e-6,
            # "filter_radius": 1.2,  # for domain size (60, 20)
            "beta_interval": 50,
            "beta_max": 128,
            "use_oc": True,
            "move": 0.02,
            "opt_compliance": True,
        },
    'simjeb': {  # Topology optimization parameters
            "max_iter": 80,
            # "max_iter": 10,
            "opt_tol": 1e-5,
            # "vol_frac": 0.08,
            # "solid_zone": lambda x: np.full(x.shape[1], False),
            # "void_zone": lambda x: np.full(x.shape[1], False),
            # "penalty": 3.0,
            "epsilon": 1e-6,
            # "filter_radius": 0.05,
            "beta_interval": 50,
            "beta_max": 128,
            "use_oc": True,
            "move": 0.02,
            "opt_compliance": True,
        }
}