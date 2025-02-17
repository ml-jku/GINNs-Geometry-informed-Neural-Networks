from __future__ import absolute_import

from .deflation import compute_tau, setSnesBounds
from .compatibility import get_deep_submat
from dolfin import *
from petsc4py import PETSc
from numpy import where, array, int32
from ufl import diag
import numpy as np

def vec(x):
    if isinstance(x, Function):
        x = x.vector()
    return as_backend_type(x).vec()

class BensonMunson(object):
    def __init__(self, problem):
        self.problem = problem

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def solver(self, problem, base, params, solver_params, prefix="", vi = None, **kwargs):
        if vi != None:
            base.snes.setType("vinewtonrsls")
            setSnesBounds(base.snes, vi)
        return base

class HintermullerItoKunisch(object):
    def __init__(self, problem):
        self.problem = problem

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def solver(self, problem, base, params, solver_params, prefix="", vi = None, **kwargs):
        if PETSc.Sys.getVersion()[0:2] < (3, 10):
            raise("This PETSc version has a bug where the linesearch is incorrectly implemented. Use PETSc version >= 3.10")
        snes = base.snes
        mesh = problem.u.function_space().mesh()
        comm = mesh.mpi_comm()

        # Needed for the definition of "plus" later
        zero = vec(Function(problem.u.function_space()))

        deflation = problem.deflation

        (lb, ub) = map(vec, self.problem.bounds_vi(problem.u.function_space(), Constant(0),params))

        z = vec(problem.u.copy(True))
        F = vec(assemble(problem.F))
        self.residualChanges = F.copy()

        self.slack_lb = z.duplicate()
        self.slack_ub = z.duplicate()

        # Grab initial active & inactive sets for residual calculation

        lb_active_dofs1   = where( ((z.array_r - lb.array_r) <=  0) )[0].astype('int32')
        lb_active_dofs2   = where( (F.array_r > 0))[0].astype('int32')
        lb_active_dofs = array(list(set(lb_active_dofs1).intersection(set(lb_active_dofs2))), dtype=int32)
        lb_active_is = PETSc.IS().createGeneral(lb_active_dofs, comm=comm)
        self.lb_active_is_old = lb_active_is

        ub_active_dofs1   = where( ((z.array_r - ub.array_r) >=  0) )[0].astype('int32')
        ub_active_dofs2   = where( (F.array_r < 0))[0].astype('int32')
        ub_active_dofs = array(list(set(ub_active_dofs1).intersection(set(ub_active_dofs2))), dtype=int32)
        ub_active_is = PETSc.IS().createGeneral(ub_active_dofs, comm=comm)
        self.ub_active_is_old = ub_active_is

        # Set initial guess of slacks to be equal to residual on active sets

        slack_lb_active = self.slack_lb.getSubVector(lb_active_is)
        F_lb = F.getSubVector(lb_active_is)
        F_lb.copy(slack_lb_active)
        self.slack_lb.restoreSubVector(lb_active_is, slack_lb_active)
        F.restoreSubVector(lb_active_is, F_lb)

        slack_ub_active = self.slack_ub.getSubVector(ub_active_is)
        F_ub = F.getSubVector(ub_active_is)
        F_ub.copy(slack_ub_active)
        slack_ub_active.scale(-1.0)
        self.slack_ub.restoreSubVector(ub_active_is, slack_ub_active)
        F.restoreSubVector(ub_active_is, F_ub)

        class MySolver(object):
            def step(iself, snes, X, F, Y):

                # X is the current guess for the solution
                # F is the residual as defined by F(u^k) where we are trying to solve F(u) = 0
                # Y is where the step is stored

                Xorig = X.copy()

                # The residual is changed in newresidual for convergence testing and
                # linesearch purposes. The following line resets the residual to whatever
                # it should be for semismooth Newton step calculation
                F.axpy(+1.0,self.residualChanges)

                J = snes.getJacobian()[0]
                snes.computeJacobian(X, J)

                # To make the notation more mathematical
                z  = X
                dz = Y

                # Make copies of the stored slack variables
                slack_lb = self.slack_lb.copy()
                slack_ub = self.slack_ub.copy()

                # Calculate the active and inactive sets
                lb_active_dofs   = where(slack_lb.array_r - (z.array_r - lb.array_r) >  0)[0].astype('int32')
                lb_active_is = PETSc.IS().createGeneral(lb_active_dofs, comm=comm)
                lb_inactive_dofs = where(slack_lb.array_r - (z.array_r - lb.array_r) <= 0)[0].astype('int32')
                lb_inactive_is = PETSc.IS().createGeneral(lb_inactive_dofs, comm=comm)
                self.lb_active_is_old = lb_active_is

                ub_active_dofs   = where(slack_ub.array_r - (ub.array_r - z.array_r) >  0)[0].astype('int32')
                ub_active_is = PETSc.IS().createGeneral(ub_active_dofs, comm=comm)
                ub_inactive_dofs = where(slack_ub.array_r - (ub.array_r - z.array_r) <= 0)[0].astype('int32')
                ub_inactive_is = PETSc.IS().createGeneral(ub_inactive_dofs, comm=comm)
                self.ub_active_is_old = ub_active_is

                inactive_dofs = array(list(set(lb_inactive_dofs).intersection(set(ub_inactive_dofs))), dtype=int32)
                inactive_is   = PETSc.IS().createGeneral(inactive_dofs, comm=comm)
                self.inactive_is_old = inactive_is

                dz_lb_active = dz.getSubVector(lb_active_is)
                dz_ub_active = dz.getSubVector(ub_active_is)
                dz_inactive  = dz.getSubVector(inactive_is)

                # Set dz where the lower bound is active.
                lb_active = lb.getSubVector(lb_active_is)
                z_lb_active = z.getSubVector(lb_active_is)
                lb_active.copy(dz_lb_active) # where lower bound is active: dz = lb
                dz_lb_active.axpy(-1.0, z_lb_active) #                      dz = lb - z
                z.restoreSubVector(lb_active_is, z_lb_active)

                # Set dz where the upper bound is active.
                ub_active = ub.getSubVector(ub_active_is)
                z_ub_active = z.getSubVector(ub_active_is)
                ub_active.copy(dz_ub_active) # where upper bound is active: dz = ub
                dz_ub_active.axpy(-1.0, z_ub_active) #                      dz = ub - z
                z.restoreSubVector(ub_active_is, z_ub_active)

                # Solve the PDE where the constraints are inactive.
                M       = J
                M_inact = get_deep_submat(M, inactive_is, inactive_is)
                M_lb    = get_deep_submat(M, inactive_is, lb_active_is)
                M_ub    = get_deep_submat(M, inactive_is, ub_active_is)

                F_inact = F.getSubVector(inactive_is)

                rhs = -F_inact.copy()
                tmp = F_inact.duplicate()

                M_lb.mult(dz_lb_active, tmp)
                rhs.axpy(-1.0, tmp)
                M_ub.mult(dz_ub_active, tmp)
                rhs.axpy(-1.0, tmp)

                ksp = PETSc.KSP().create(comm=comm)
                ksp.setOperators(M_inact)
                ksp.setType("preonly")
                ksp.pc.setType("lu")
                ksp.pc.setFactorSolverType("mumps")
                ksp.setFromOptions()
                ksp.setUp()
                ksp.solve(rhs, dz_inactive)

                del rhs
                del M_ub, M_lb, M_inact


                F.restoreSubVector(inactive_is, F_inact)
                lb.restoreSubVector(lb_active_is, lb_active)
                ub.restoreSubVector(ub_active_is, ub_active)
                dz.restoreSubVector(inactive_is,  dz_inactive)


                # Now update inactive set of the slacks
                slack_lb_inactive = slack_lb.getSubVector(inactive_is)
                slack_ub_inactive = slack_ub.getSubVector(inactive_is)
                slack_lb_inactive.array[:] = 0
                slack_ub_inactive.array[:] = 0
                slack_lb.restoreSubVector(inactive_is, slack_lb_inactive)
                slack_ub.restoreSubVector(inactive_is, slack_ub_inactive)

                # Active set of slacks are updated when calculating the new residual

                # Store active sets of residual for later use in updating slacks
                F_lb = F.getSubVector(lb_active_is)
                self.Flb = F_lb.copy()
                F.restoreSubVector(lb_active_is, F_lb)

                F_ub = F.getSubVector(ub_active_is)
                self.Fub = F_ub.copy()
                F.restoreSubVector(ub_active_is, F_ub)

                # Store Hessian used later to update slacks
                self.M = M.copy()

                dz.restoreSubVector(ub_active_is, dz_ub_active)
                dz.restoreSubVector(lb_active_is, dz_lb_active)

                # dz was only a pointer, so Y is up to date.
                Y.scale(-1.0) # PETsc has weird conventions....

                # Update stored slacks
                slack_lb.copy(self.slack_lb)
                slack_ub.copy(self.slack_ub)

                # Compute resizing due to deflation
                tau = compute_tau(deflation, problem.u, Y, None)
                Y.scale(tau)

                # Store current iterate and step for later convergence testing
                self.Xold = X.copy()
                self.Y = Y.copy()
        (f, (fenicsresidual, args, kargs)) = snes.getFunction()

        def plus(X, out):
            out.pointwiseMax(X, zero)


        def newresidual(snes, X, F):

            # This is a hack to alter the SNES norm calculated for the convergence
            # I alter the residual which is used to calculate the norm. I keep the old
            # residual in residualChanged and change the residual back to what it should
            # be in step in order to do the correct semismooth Newton step.
            fenicsresidual(snes, X, F)

            # Here we update the active set of the slacks
            # We also (optionally) ensure the correct update to the state active sets
            if snes.getIterationNumber() > 0:
                X_old = self.Xold
                lb_active_is = self.lb_active_is_old
                ub_active_is = self.ub_active_is_old

                lb_active = lb.getSubVector(lb_active_is)
                ub_active = ub.getSubVector(ub_active_is)
                X_active_lb = X.getSubVector(lb_active_is)
                X_active_ub = X.getSubVector(ub_active_is)

                # Here we ensure the active set goes to the bound irrespective of
                # the linesearch stepsize
                lb_active.copy(X_active_lb)
                ub_active.copy(X_active_ub)

                X.restoreSubVector(lb_active_is, X_active_lb)
                X.restoreSubVector(ub_active_is, X_active_ub)
                lb.restoreSubVector(lb_active_is, lb_active)
                ub.restoreSubVector(ub_active_is, ub_active)

                # Update residual due to change of active sets in state
                fenicsresidual(snes, X, F)

                slack_lb_active = self.slack_lb.getSubVector(lb_active_is)
                slack_ub_active = self.slack_ub.getSubVector(ub_active_is)

                dz = X.copy()
                dz.axpy(-1.0, X_old)

                tmp = dz.duplicate()
                self.M.mult(dz, tmp)

                tmp_lb_active = tmp.getSubVector(lb_active_is)
                self.Flb.copy(slack_lb_active)
                slack_lb_active.axpy(+1.0, tmp_lb_active)
                tmp.restoreSubVector(lb_active_is, tmp_lb_active)

                tmp_ub_active = tmp.getSubVector(ub_active_is)
                self.Fub.copy(slack_ub_active)
                slack_ub_active.scale(-1.0)
                slack_ub_active.axpy(-1.0, tmp_ub_active)
                tmp.restoreSubVector(ub_active_is, tmp_ub_active)


                self.slack_lb.restoreSubVector(lb_active_is, slack_lb_active)
                self.slack_ub.restoreSubVector(ub_active_is, slack_ub_active)

            # Store original residual
            F.copy(self.residualChanges)

            # For ease of mathematical notation
            z = X

            # If solution has already been found, we want the calculated norm to be
            # zero. However, since we do not store the slacks, our original guesses
            # for the slacks might produce a nonzero residual. Using slacks of zero
            # in the first calculation ensures that the convergence is correctly calculated
            if snes.getIterationNumber() > 0:
                slack_lb = self.slack_lb
                slack_ub = self.slack_ub
            else:
                slack_lb = X.duplicate()
                slack_ub = X.duplicate()

            # This is our choice of convergence testing. I use the idea of the
            # a mix complementarity problem (MCP) to define if my residuals are correct.

            # We want
            #         If z = lb       then   F \geq 0
            #         If lb < z < ub  then   F = 0
            #         If z = ub       then   F \leq 0

            # Hence below I zero the contributions of the residual associated with
            # the active set of lb where F is larger than zero and I zero the
            # contributions of the residual associated with the active set of ub
            # where F is smaller than zero.
            #
            # F_lb = F.getSubVector(self.lb_active_is_old)
            # zero_ = F_lb.duplicate()
            # F_lb.pointwiseMin(F_lb, zero_)
            # F.restoreSubVector(self.lb_active_is_old, F_lb)
            #
            #
            # F_ub = F.getSubVector(self.ub_active_is_old)
            # zero_ = F_ub.duplicate()
            # F_ub.pointwiseMax(F_ub, zero_)
            # F.restoreSubVector(self.ub_active_is_old, F_ub)

            # Alternatively also instead sum the slacks i.e.
            F.axpy(-1.0, slack_lb)            # F = J'(u) - slack_lb
            F.axpy(+1.0, slack_ub)            # F = J'(u) + slack_ub - slack_lb

            tmp_state = F.duplicate()
            tmp_c1 = F.duplicate()
            tmp_c2 = F.duplicate()
            F_lb = F.duplicate()
            F_ub = F.duplicate()

            # Now set the residual for lb.

            z.copy(tmp_c1)             # tmp1 = u
            tmp_c1.axpy(-1.0, lb)      # tmp1 = u - a
            tmp_c1.scale(-1.0)         # tmp1 = -(u - a)
            tmp_c1.axpy(+1.0, slack_lb)   # tmp1 = slack_lb - (u - a)
            plus(tmp_c1, tmp_c2)       # tmp2 = (slack_lb - (u - a))_+
            tmp_c2.scale(-1.0)         # tmp2 = - (slack_lb - (u - a))_+
            tmp_c2.axpy(+1.0, slack_lb)   # tmp2 = slack_lb - (slack_lb - (u - a))_+
            tmp_c2.copy(F_lb)

            # Now set the residual for ub.

            ub.copy(tmp_c1)            # tmp1 = b
            tmp_c1.axpy(-1.0, z)       # tmp1 = b - u
            tmp_c1.scale(-1.0)         # tmp1 = -(b - u)
            tmp_c1.axpy(+1.0, slack_ub)   # tmp1 = mu - (b - u)
            plus(tmp_c1, tmp_c2)       # tmp2 = (mu - (b - u))_+
            tmp_c2.scale(-1.0)         # tmp2 = - (mu - (b - u))_+
            tmp_c2.axpy(+1.0, slack_ub)   # tmp2 = mu - (mu - (b - u))_+
            tmp_c2.copy(F_ub)

            # For testing
            # print("|F|: ", F.norm(PETSc.NormType.NORM_2))
            # print("|F_lb|: ", F_lb.norm(PETSc.NormType.NORM_2))
            # print("|F_ub|: ", F_ub.norm(PETSc.NormType.NORM_2))

            F.axpy(+1.0, F_lb)
            F.axpy(+1.0, F_ub)

            norm = F.norm(PETSc.NormType.NORM_2)
            # Store difference in the residual for the step calculation where it is reset
            self.residualChanges.axpy(-1.0, F)

        # Activate if you want to save the iterates into a pvd
        def newmonitor(snes, it, residual):
             z = snes.getSolution()
             z_vec = PETScVector(z)
             pvd.vector()[:] = z_vec
             pvd_state << pvd

        snes.setType("python")
        snes.setPythonContext(MySolver())
        snes.setFunction(newresidual, f)
        # snes.setMonitor(newmonitor)

        return base

class ProjectedNewton(object):
    def __init__(self, problem):
        self.problem = problem

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def solver(self, problem, base, params, solver_params, prefix="", vi = None, **kwargs):
        # base = self.problem.solver(problem, params, solver_params, prefix=prefix, **kwargs)
        snes = base.snes

        # u_dvec = as_backend_type(problem.u.vector())
        mesh = problem.u.function_space().mesh()
        comm = mesh.mpi_comm()

        deflation = problem.deflation
        is_rho = PETSc.IS().createGeneral(problem.u.function_space().sub(0).dofmap().dofs(), comm=comm)
        (f, (fenicsresidual, args, kargs)) = snes.getFunction()

        def newmonitor(snes, it, residual):
             x = snes.getSolution()
             # comm = x.function_space().mesh().mpi_comm()

             rho = x.getSubVector(is_rho)

             zero = rho.copy(); zero.array[:] = 0.0
             one  = rho.copy(); one.array[:] = 1.0

             # print("max/min rho before clipping (%s, %s)" %(min(rho.array), max(rho.array)))
             rho.pointwiseMax(rho, zero)
             rho.pointwiseMin(rho, one)
             # print("max/min rho after clipping (%s, %s)" %(min(rho.array), max(rho.array)))

             x.restoreSubVector(is_rho, rho)
             # print("max/min x after clipping (%s, %s)" %(min(x.array), max(x.array)))

        def plus(X, out):
            out.pointwiseMax(X, zero)

        def newresidual(snes, X, F):
            fenicsresidual(snes, X, F)

            rho = X.getSubVector(is_rho)
            # print("max/min rho after clipping (%s, %s)" %(min(rho.array), max(rho.array)))
            lb_dofs   = where(rho.array_r <= 0.)[0].astype('int32')
            is_lb = PETSc.IS().createGeneral(lb_dofs, comm=comm)
            ub_dofs   = where(rho.array_r >= 1.)[0].astype('int32')
            is_ub = PETSc.IS().createGeneral(ub_dofs, comm=comm)

            # print("|F|: ", F.norm(PETSc.NormType.NORM_2))
            F_lb = F.getSubVector(is_lb)
            F_ub = F.getSubVector(is_ub)

            zero = F_lb.duplicate(); zero.array[:] = 0.0
            F_lb.pointwiseMin(F_lb, zero)
            # F_lb.pointwiseMax(F_lb, zero)
            zero = F_ub.duplicate(); zero.array[:] = 0.0
            F_ub.pointwiseMax(F_ub, zero)
            # F_ub.pointwiseMin(F_ub, zero)
            X.restoreSubVector(is_rho, rho)
            # print("|F_lb|: ", F_lb.norm(PETSc.NormType.NORM_2))
            # print("|F_ub|: ", F_ub.norm(PETSc.NormType.NORM_2))
            F.restoreSubVector(is_ub, F_ub)
            F.restoreSubVector(is_lb, F_lb)
            # print("|F|: ", F.norm(PETSc.NormType.NORM_2))
            res = F.norm(PETSc.NormType.NORM_2)

        # snes.setType("python")
        # snes.setPythonContext(MySolver())
        snes.setFunction(newresidual, f)
        snes.setMonitor(newmonitor)
        return base


def fb(a, b):
    """
    Fischer--Burmeister merit function.
    """
    return sqrt(a**2 + b**2) - a - b
