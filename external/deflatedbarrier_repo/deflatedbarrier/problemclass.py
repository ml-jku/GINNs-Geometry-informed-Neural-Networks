from dolfin import *
import dolfin
from .deflation import ShiftedDeflation
from .nonlinearsolver import SNUFLSolver
from .nonlinearproblem import GeneralProblem
from .misc import soln, plus, MorYos, create_output_folder, inertia_switch, report_profile
from .prediction import feasibletangent, tangent, secant, nothing
from petsc4py import PETSc

import numpy as np


# File containing the PrimalInteriorPoint and PrimalDualInteriorPoint classes

class PrimalInteriorPoint(object):
    def function_space(self, mesh):
        raise NotImplementedError

    def mesh(self, comm):
        raise NotImplementedError

    def coarse_meshes(self, comm):
        return None

    def expected_inertia(self):
        return None

    def initial_guesses(self, V, params):
        raise NotImplementedError

    def lagrangian(self, u, params):
        raise NotImplementedError

    def penalty(self, z, lb, ub, mu, params):
        rho = split(z)[0]
        return -mu*ln(rho - lb)*dx  - mu*ln(ub - rho)*dx

    def epsilon(self, z):
        return NotImplementedError

    def residual(self, u, v, lb, ub, mu, params):
        J = self.lagrangian(u, params) + self.penalty(u, lb, ub, mu, params)
        F = derivative(J, u, v)
        return F

    def jacobian(self, F, state, params, test, trial):
        return derivative(F, state, trial)

    def boundary_conditions(self, V, params):
        raise NotImplementedError

    def infeasibility(self, z, lb, ub, mu, params):
        u = split(z)[0]
        return plus(lb-u)**2*dx + plus(u - ub)**2*dx

    def squared_norm(self, a, b, params):
        arho = split(a)[0]
        brho = split(b)[0]
        return (inner(arho - brho, arho - brho)*dx)

    def solver_parameters(self, mu, branch, task, params):
        raise NotImplementedError

    def save_pvd(self, pvd, z, mu):
        if float(mu) == 0.0:
            rho_ = z.split(deepcopy=True)[0]
            rho_.rename("Control", "Control")
            pvd << rho_


    def save_solution(self, mesh, z, mu, params, k, branch, pathfile):
        print(f'save solution')
        
        
        comm = mesh.mpi_comm()
        hmin = mesh.hmin()
        h5 = HDF5File(comm, pathfile + "/%s.xml.gz" %branch, "w")
        h5.write(z, "/guess")
        del h5
        
        # [AR 8.1.2025] modified to save the density field in a more accessible format
        # rho = z.vector().get_local()[:7701]
        V0 = z.function_space().split()[0]
        # V_values = z.function_space().sub(0)
        rho = z.vector()[V0.dofmap().dofs()]
        coords = V0.collapse().tabulate_dof_coordinates()
        np.save(pathfile + "/%s_rho.npy" %branch, rho)
        np.save(pathfile + "/%s_coords.npy" %branch, coords)
        
        return

    def bounds(self, mesh, mu, params):
        ep = 1e-5
        return (Constant(0.0-ep), Constant(1.0+ep))

    def bounds_vi(self, Z, mu, params):
        return None

    def update_mu(self, u, mu, iters, k, k_mu_old, params):
        # rules of IPOPT DOI: 10.1007/s10107-004-0559-y
        etol = 1e-15
        k_mu = 0.7
        theta_mu = 1.5
        next_mu = max(etol/10., min(k_mu*mu, mu**theta_mu))
        return next_mu

    def deflation_operator(self, params):
        return ShiftedDeflation(2, 1, self.squared_norm)

    def number_solutions(self, mu, params):
        return 1

    def compute_stability(self, mu, params, lb, ub, branch, z, v, w, Z, bcs, J):
        trial = w
        test  = v
        comm = Z.mesh().mpi_comm()
        # a dummy linear form, needed to construct the SystemAssembler
        b = inner(Function(Z), test)*dx  # a dummy linear form, needed to construct the SystemAssembler

        # Build the LHS matrix
        A = PETScMatrix(comm)

        asm = SystemAssembler(J, b, bcs)
        asm.assemble(A)

        pc = PETSc.PC().create(comm)
        pc.setOperators(A.mat())
        pc.setType("cholesky")
        if PETSc.Sys.getVersion()[0:2] < (3, 9):
            pc.setFactorSolverPackage("mumps")
        else:
            pc.setFactorSolverType("mumps")
        pc.setUp()

        Factor = pc.getFactorMatrix()
        (neg, zero, pos) = Factor.getInertia()
        inertia  = [neg, zero, pos]
        expected_dim = 0

        # Nocedal & Wright, theorem 16.3
        if neg == expected_dim:
            is_stable = True
        else:
            is_stable = False

        d = {"stable": is_stable}
        return inertia

    def solver(self, problem, base, params, solver_params, prefix="", vi = None, dm=None):
        return base

    def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
        return nothing(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)

    def cost(self, z, params):
        return assemble(self.lagrangian(self, z, params))

    def volume_constraint(self, params):
        raise NotImplementedError

class PrimalDualInteriorPoint(object):

    def finite_element(self, mesh):
        raise NotImplementedError

    def function_space(self, mesh):
        Ze = self.finite_element(mesh)
        Z = FunctionSpace(mesh, Ze)
        return Z

    def mesh(self, comm):
        raise NotImplementedError

    def coarse_meshes(self, comm):
        return None

    def initial_guesses(self, Z, params):
        raise NotImplementedError

    def lagrangian(self, z, params):
        raise NotImplementedError

    def penalty(self, z, w, lb, ub, mu, params):
        u = split(z)[0]
        v = split(w)[0]
        lb_u = split(z)[-2]
        ub_u = split(z)[-1]
        lb_v = split(w)[-2]
        ub_v = split(w)[-1]
        P = - inner(lb_u,v)*dx + inner(ub_u,v)*dx \
            + (ub - u)*ub_u*ub_v*dx - mu*ub_v*dx \
            + (u - lb)*lb_u*lb_v*dx - mu*lb_v*dx
        return P

    def infeasibility(self, z, lb, ub, mu, params):
        u = split(z)[0]
        return plus(lb-u)**2*dx + plus(u - ub)**2*dx

    def residual(self, z, w, lb, ub, mu, params):
        J = self.lagrangian(z, params)
        F = derivative(J, z, w)
        R = F + self.penalty(z, w, lb, ub, mu, params)
        return R


    def boundary_conditions(self, Z, params):
        raise NotImplementedError

    def jacobian(self, F, state, params, test, trial):
        return derivative(F, state, trial)

    def bounds(self, mesh, mu, params):
        raise NotImplementedError

    def squared_norm(self, u, v, params):
        return inner(u - v, u - v)*dx

    def solver_parameters(self, mu, branch, task, params):
        raise NotImplementedError

    def save_pvd(self, pvd, z, mu):
        if float(mu) == 0.0:
            rho_ = z.split(deepcopy=True)[0]
            rho_.rename("Control", "Control")
            pvd << rho_

    def save_solution(self, mesh, z, mu, params, k, branch, pathfile):
        comm = mesh.mpi_comm()
        hmin = mesh.hmin()
        h5 = HDF5File(comm,pathfile + "/%s.xml.gz" %branch, "w")
        h5.write(z, "/guess")
        del h5
        return

    def update_mu(self, u, mu, iters, k, k_mu_old, params):
        # rules of IPOPT DOI: 10.1007/s10107-004-0559-y
        etol = 1e-15
        k_mu = 0.7
        theta_mu = 1.5
        next_mu = max(etol/10., min(k_mu*mu, mu**theta_mu))
        return next_mu


    def deflation_operator(self, params):
        return ShiftedDeflation(2, 1, self.squared_norm)

    def nullspace(self, V, params):
        return None

    def number_solutions(self, mu, params):
        #return float("inf")
        if float(mu) > 0:
            return 1
        else:
            return 10

    def compute_stability(self, mu, params, lb, ub, branch, z, v, w, Z, bcs, J):
        # trial = w
        # test  = v
        test = TestFunction(Z)
        comm = Z.mesh().mpi_comm()
        # a dummy linear form, needed to construct the SystemAssembler
        b = inner(Function(Z), test)*dx  # a dummy linear form, needed to construct the SystemAssembler

        # Build the LHS matrix
        A = PETScMatrix(comm)

        asm = SystemAssembler(J, b, bcs)
        asm.assemble(A)

        pc = PETSc.PC().create(comm)
        pc.setOperators(A.mat())
        pc.setType("cholesky")
        if PETSc.Sys.getVersion()[0:2] < (3, 9):
            pc.setFactorSolverPackage("mumps")
        else:
            pc.setFactorSolverType("mumps")
        pc.setUp()

        Factor = pc.getFactorMatrix()
        (neg, zero, pos) = Factor.getInertia()
        inertia  = [neg, zero, pos]
        expected_dim = 0

        # Nocedal & Wright, theorem 16.3
        if neg == expected_dim:
            is_stable = True
        else:
            is_stable = False

        d = {"stable": is_stable}
        return inertia

    def bounds_vi(self, Z, mu, params):
        return None

    def solver(problem, base, params, solver_params, prefix="", vi = None, dm=None):
        return base

    def volume_constraint(self, params):
        raise NotImplementedError

    def cost(self, z, params):
        return assemble(self.lagrangian(self, z, params))

    def predictor(self, problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
        return nothing(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint)
