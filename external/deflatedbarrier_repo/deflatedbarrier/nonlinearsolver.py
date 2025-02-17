from __future__ import absolute_import


from petsc4py import PETSc
import ufl.algorithms
import numpy
from dolfin import as_backend_type, PETScVector, PETScMatrix, \
    MixedElement, VectorElement, Function, FunctionSpace, \
    SystemAssembler, Form
from petsc4py import PETSc
# dolfin lacks a high-level snes frontend like Firedrake,
# so we're going to put one here and build up what we need
# to make things happen.
class SNUFLSolver(object):
    def __init__(self, problem, prefix="", solver_parameters={}, **kwargs):
        self.problem = problem
        u = problem.u
        self.u_dvec = as_backend_type(u.vector())
        self.u_pvec = self.u_dvec.vec()

        comm = u.function_space().mesh().mpi_comm()
        self.comm = comm
        snes = PETSc.SNES().create(comm=comm)
        snes.setOptionsPrefix(prefix)

        # Fix the worst defaults in PETSc
        opts = PETSc.Options()
        if "snes_linesearch_type" not in solver_parameters:
            opts[prefix + "snes_linesearch_type"] = "basic"
        # if "snes_divergence_tolerance" not in solver_parameters:
        #     opts[prefix + "snes_divergence_tolerance"] = -1
        if "snes_stol" not in solver_parameters:
            opts[prefix + "snes_stol"] = 0.0

        # No backwards compatibility in PETSc
        if PETSc.Sys.getVersion()[0:2] >= (3, 9):
            if "pc_factor_mat_solver_package" in solver_parameters:
                opts[prefix + "pc_factor_mat_solver_type"] = solver_parameters["pc_factor_mat_solver_package"]
                del solver_parameters["pc_factor_mat_solver_package"]
        else:
            if "pc_factor_mat_solver_type" in solver_parameters:
                opts[prefix + "pc_factor_mat_solver_package"] = solver_parameters["pc_factor_mat_solver_type"]
                del solver_parameters["pc_factor_mat_solver_type"]

        # set the petsc options from the solver_parameters
        for k in solver_parameters:
            opts[prefix + k] = solver_parameters[k]

        (J, F, bcs, P) = (problem.J, problem.F, problem.bcs, problem.P)

        if hasattr(problem, 'problem'):
            self.ass = problem.problem.assembler(J, F, u, bcs)
            if P is not None:
                self.Pass = problem.problem.assembler(P, F, u, bcs)
        else:
            self.ass = SystemAssembler(J, F, bcs)
            if P is not None:
                self.Pass = SystemAssembler(P, F, bcs)

        self.b = self.init_residual()
        snes.setFunction(self.residual, self.b.vec())
        self.A = self.init_jacobian()
        self.P = self.init_preconditioner(self.A)
        snes.setJacobian(self.jacobian, self.A.mat(), self.P.mat())
        # why isn't this done in setJacobian?
        snes.ksp.setOperators(self.A.mat(), self.P.mat())

        # If the user wants to correctly compute dual norms, let them
        if hasattr(problem, 'problem') and "objective" in problem.problem.__class__.__dict__:
            snes.setObjective(self.objective)

        # Workaround for bug in Cray PETSc on ARCHER
        if "pc_type" in solver_parameters:
            snes.ksp.pc.setType(solver_parameters["pc_type"])

        dm = kwargs.get("dm", None)
        if dm is not None:
            snes.setDM(dm) # dm is passed via kwargs to make it easy
                           # to keep interface consistent with firedrake
        snes.setFromOptions()
        self.snes = snes

    def init_jacobian(self):
        A = PETScMatrix(self.comm)
        self.ass.init_global_tensor(A, Form(self.problem.J))
        return A

    def init_residual(self):
        b = as_backend_type(
            Function(self.problem.u.function_space()).vector()
        )
        return b

    def init_preconditioner(self, A):
        if self.problem.P is None:
            return A
        P = PETScMatrix(self.comm)
        self.Pass.init_global_tensor(P, Form(self.problem.P))
        return P

    def update_x(self, x):
        """Given a PETSc Vec x, update the storage of our
           solution function u."""

        x.copy(self.u_pvec)
        self.u_dvec.update_ghost_values()

    def residual(self, snes, x, b):
        self.update_x(x)
        b_wrap = PETScVector(b)
        self.ass.assemble(b_wrap, self.u_dvec)

    def jacobian(self, snes, x, A, P):
        self.update_x(x)
        A_wrap = PETScMatrix(A)
        P_wrap = PETScMatrix(P)
        self.ass.assemble(A_wrap)
        if self.problem.P is not None:
            self.Pass.assemble(P_wrap)

    def objective(self, snes, b):
        F = self.problem.F
        v = ufl.algorithms.extract_arguments(F)[0]
        r = self.problem.problem.objective(self.problem.F, self.problem.u, v)
        return r

    def solve(self):
        # Need a copy for line searches etc. to work correctly.
        x = self.problem.u.copy(deepcopy=True)
        xv = as_backend_type(x.vector()).vec()

        try:
            self.snes.solve(None, xv)
        except:
            import traceback
            traceback.print_exc()
            pass
