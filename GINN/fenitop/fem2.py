import numpy as np
import ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import (functionspace, Function, Constant,
                         dirichletbc, locate_dofs_topological, ElementMetaData)
import basix.ufl as bufl

from .utility import create_mechanism_vectors
from .utility import LinearProblem


def form_fem2(fem, opt):
    """Form an FEA problem."""
    # Function spaces and functions
    mesh = fem["mesh"]
    ed = ElementMetaData(*("CG", 1))
    ufl_e = bufl.element(ed.family, mesh.basix_cell(), ed.degree, shape=(mesh.geometry.dim,))
    V = functionspace(mesh, ufl_e)
    S0 = functionspace(mesh, ("DG", 0))
    S = functionspace(mesh, ("CG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_field = Function(V)  # Displacement field
    lambda_field = Function(V)  # Adjoint variable field
    rho_field = Function(S0)  # Density field
    rho_phys_field = Function(S)  # Physical density field

    # Material interpolation
    E0, nu = fem["young's modulus"], fem["poisson's ratio"]
    p, eps = opt["penalty"], opt["epsilon"]
    E = (eps + (1-eps)*rho_phys_field**p) * E0
    _lambda, mu = E*nu/(1+nu)/(1-2*nu), E/(2*(1+nu))  # Lame constants

    # Kinematics
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):  # 3D or plane strain
        return 2*mu*epsilon(u) + _lambda*ufl.tr(epsilon(u))*ufl.Identity(len(u))

    # Boundary conditions
    dim = mesh.topology.dim
    fdim = dim - 1

    V1, _ = V.sub(0).collapse()
    u_D = Function(V1)
    u_D.x.array[:] = 0.

    V2, _ = V.sub(1).collapse()
    u_D2 = Function(V2)
    u_D2.x.array[:] = 0.
    #boundary_facets = np.flatnonzero(mesh.exterior_facet_indices(domain.topology))
    disp_facets1 = locate_entities_boundary(mesh, fdim, fem["disp_bc1"])
    disp_facets2 = locate_entities_boundary(mesh, fdim, fem["disp_bc2"])

    boundary_dofs_b1 = locate_dofs_topological((V.sub(0), V1), fdim, disp_facets1)
    boundary_dofs_b2 = locate_dofs_topological((V.sub(1), V2), fdim, disp_facets2)

    bc1 = dirichletbc(u_D, boundary_dofs_b1, V.sub(1))
    bc2 = dirichletbc(u_D2, boundary_dofs_b2, V.sub(0))

    # Boundary conditions
    dim = mesh.topology.dim
    fdim = dim - 1
    disp_facets = locate_entities_boundary(mesh, fdim, fem["disp_bc1"])
    bc = dirichletbc(Constant(mesh, np.full(dim, 0.0)),
                    locate_dofs_topological(V, fdim, disp_facets), V)

    tractions, facets, markers = [], [], []
    for marker, (traction, traction_bc) in enumerate(fem["traction_bcs"]):
        tractions.append(Constant(mesh, np.array(traction, dtype=float)))
        current_facets = locate_entities_boundary(mesh, fdim, traction_bc)
        facets.extend(current_facets)
        markers.extend([marker,]*len(current_facets))
    facets = np.array(facets, dtype=np.int32)
    markers = np.array(markers, dtype=np.int32)
    _, unique_indices = np.unique(facets, return_index=True)
    facets, markers = facets[unique_indices], markers[unique_indices]
    sorted_indices = np.argsort(facets)
    facet_tags = meshtags(mesh, fdim, facets[sorted_indices], markers[sorted_indices])

    metadata = {"quadrature_degree": fem["quadrature_degree"]}
    dx = ufl.Measure("dx", metadata=metadata)
    ds = ufl.Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_tags)
    b = Constant(mesh, np.array(fem["body_force"], dtype=float))
    opt["total_volume"] = Constant(mesh, 1.0)*dx
    
        # Establish the equilibrium and adjoint equations
    lhs = ufl.inner(sigma(u), epsilon(v))*dx
    rhs = ufl.dot(b, v)*dx
    for marker, t in enumerate(tractions):
        rhs += ufl.dot(t, v)*ds(marker)
    if opt["opt_compliance"]:
        spring_vec = opt["l_vec"] = None
    else:
        spring_vec, opt["l_vec"] = create_mechanism_vectors(
            V, opt["in_spring"], opt["out_spring"])
    linear_problem = LinearProblem(u_field, lambda_field, lhs, rhs, opt["l_vec"],
                                spring_vec, [bc1, bc2], fem["petsc_options"])
    
    # Define optimization-related variables
    opt["f_int"] = ufl.inner(sigma(u_field), epsilon(v))*dx
    opt["compliance"] = ufl.inner(sigma(u_field), epsilon(u_field))*dx
    opt["volume"] = rho_phys_field*dx
    opt["total_volume"] = Constant(mesh, 1.0)*dx

    return linear_problem, u_field, lambda_field, rho_field, rho_phys_field
