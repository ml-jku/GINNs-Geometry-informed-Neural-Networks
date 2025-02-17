"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

import numpy as np
import ufl
from dolfinx import la
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import create_matrix, assemble_matrix
from petsc4py import PETSc


class DensityFilter():
    def __init__(self, comm, rho, rho_tilde, R=1.0, petsc_options={}):
        """Construct a PDE filter."""
        # Initialization
        S0, S = rho.function_space, rho_tilde.function_space
        u0, u = ufl.TrialFunction(S0), ufl.TrialFunction(S)
        v, self.af = ufl.TestFunction(S), Function(S)

        self.rho, self.rho_tilde = rho, rho_tilde
        self.rho_tilde_wrap = la.create_petsc_vector_wrap(self.rho_tilde.x)
        self.af_wrap = la.create_petsc_vector_wrap(self.af.x)
        self.vec_s0, self.vec_s = rho.x.petsc_vec.copy(), rho_tilde.x.petsc_vec.copy()

        # Construct Kf and T matrices based on the Helmholtz PDE
        dx = ufl.Measure("dx", metadata={"quadrature_degree": 2})
        Kf_expr = (R**2*ufl.dot(ufl.grad(u), ufl.grad(v)) + u*v)*dx
        T_expr = u0*v*dx
        Kf_form, T_form = form(Kf_expr), form(T_expr)
        Kf_mat, self.T_mat = create_matrix(Kf_form), create_matrix(T_form)

        # Construct a filtering solver
        self.solver = PETSc.KSP().create(comm)
        self.solver.setOperators(Kf_mat)
        prefix = f"filter_solver_{id(self)}"
        self.solver.setOptionsPrefix(prefix)

        # Apply PETSc options
        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        Kf_mat.setOptionsPrefix(prefix)
        Kf_mat.setFromOptions()

        # Assemble Kf and T matrices
        assemble_matrix(Kf_mat, Kf_form)
        Kf_mat.assemble()
        assemble_matrix(self.T_mat, T_form)
        self.T_mat.assemble()
        self.T_mat_transpose = self.T_mat.copy()
        self.T_mat_transpose.transpose()

    def forward(self):
        """Compute the filtered variables."""
        self.T_mat.mult(self.rho.x.petsc_vec, self.vec_s)
        self.solver.solve(self.vec_s, self.rho_tilde_wrap)
        self.rho_tilde.x.scatter_forward()
        return self.rho_tilde

    def backward(self, sf_vectors):
        """Recover the sensitivities."""
        values = []
        for sf in sf_vectors:
            if sf is not None:
                self.solver.solve(sf, self.af_wrap)
                self.af.x.scatter_forward()
                self.T_mat_transpose.mult(self.af.x.petsc_vec, self.vec_s0)
                values.append(self.vec_s0.array.copy())
            else:
                values.append(None)
        return values


class Heaviside():
    def __init__(self, rho_phys):
        self.rho_phys = rho_phys

    def forward(self, beta, eta=0.5):
        denominator = np.tanh(beta*eta) + np.tanh(beta*(1-eta))
        self.drho = beta*(1-np.tanh(beta*(self.rho_phys.x.petsc_vec-eta))**2) / denominator
        self.rho_phys.x.petsc_vec.array[:] = (
            np.tanh(beta*eta)+np.tanh(beta*(self.rho_phys.x.petsc_vec-eta))) / denominator
        self.rho_phys.x.scatter_forward()

    def backward(self, vectors):
        for vector in vectors:
            if vector is not None:
                vector.array *= self.drho
