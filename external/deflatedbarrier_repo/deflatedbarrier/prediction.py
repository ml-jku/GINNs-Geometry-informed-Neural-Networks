import dolfin as backend
from ufl import derivative
from .deflation import *
from .mlogging import *
import numpy as np
from copy import deepcopy
from .visolvers import BensonMunson

def feasibletangent(problem, solution, v, w, oldmu, newmu, k, params, task, vi, hint=None):

    if k == 1:
        return (hint, 0, 0)

    info_green("Attempting to find feasible tangent predictor")
    coldmu = backend.Constant(float(oldmu))
    chgmu = backend.Constant(float(newmu)-float(oldmu))
    rho = solution.split(deepcopy=True)[0]
    Z = solution.function_space()

    (lb, ub)  = problem.bounds(Z, newmu, params)


    if vi != None:
        # If we're dealing with a VI, we need to enforce the appropriate
        # bound constraints on the update problem. Essentially, we need
        # that
        # lb <= u + du <= ub
        # so
        # lb - u <= du <= ub - u
        state = solution
        class FixTheBounds(object):
            def bounds_vi(self, Z, newmu, params):
                (lb_vi, ub_vi) = problem.bounds_vi(Z, newmu, params)
                lb_vi.vector().axpy(-1.0, state.vector())
                ub_vi.vector().axpy(-1.0, state.vector())

                return (lb_vi, ub_vi)

            def __getattr__(self, attr):
                return getattr(problem, attr)

        newproblem = FixTheBounds()
        vi_du = newproblem.bounds_vi(Z, newmu, params)
    else:
        # newproblem = problem
        vi_du = None

    # FIXME: cache the symbolic calculation once, it can be expensive sometimes
    du = backend.Function(Z)
    #du = solution.copy(deepcopy=True)

    F = problem.residual(solution, v, lb, ub, coldmu, params)
    G = derivative(F, solution, du) + derivative(F, coldmu, chgmu)
    J = problem.jacobian(G, du, params, v, w)

    # FIXME: figure out if the boundary conditions depend on
    # the parameters, and set the boundary conditions on the update
    dubcs = problem.boundary_conditions(Z, newmu)
    [dubc.homogenize() for dubc in dubcs]

    # dm = problem._dm

    # FIXME: make this optional
    # teamno = 0

    # FIXME: there's probably a more elegant way to do this.
    # Or should we use one semismooth Newton step? After all
    # we already have a Newton linearisation.
    sp = problem.solver_parameters(float(oldmu), 0, task, params)
    (success, iters, liters) = newton(G, J, du, dubcs,
                              problem.solver,
                              params,
                              sp,
                              None, None, problem._dm, vi_du)

    hint = [None, float(oldmu)]
    if not success:
        # do nothing
        return (hint, iters, liters)
    else:
        solution.assign(solution + du)
        info_green("Found feasible tangent predictor")
        return (hint, iters, liters)

def tangent(problem, solution, v, w, oldmu, newmu, k, params, task, vi, hint=None):

    if k == 1:
        return (hint, 0, 0)

    info_green("Attempting tangent linearisation")
    coldmu = backend.Constant(float(oldmu))
    chgmu = backend.Constant(float(newmu)-float(oldmu))
    rho = solution.split(deepcopy=True)[0]
    Z = solution.function_space()

    (lb, ub)  = problem.bounds(Z, newmu, params)

    vi_du = None
    # FIXME: cache the symbolic calculation once, it can be expensive sometimes
    du = backend.Function(Z)
    #du = solution.copy(deepcopy=True)

    F = problem.residual(solution, v, lb, ub, coldmu, params)
    G = derivative(F, solution, du) + derivative(F, coldmu, chgmu)
    J = problem.jacobian(G, du, params, v, w)

    # FIXME: figure out if the boundary conditions depend on
    # the parameters, and set the boundary conditions on the update
    dubcs = problem.boundary_conditions(Z, newmu)
    [dubc.homogenize() for dubc in dubcs]

    # dm = problem._dm

    # FIXME: make this optional
    # teamno = 0

    # FIXME: there's probably a more elegant way to do this.
    # Or should we use one semismooth Newton step? After all
    # we already have a Newton linearisation.
    newproblem = BensonMunson(problem)
    sp = problem.solver_parameters(float(oldmu), 0, task, params)
    (success, iters, liters) = newton(G, J, du, dubcs,
                              newproblem.solver,
                              params,
                              sp,
                              None, None, problem._dm, vi_du)

    hint = [None, float(oldmu)]
    if not success:
        #do nothing
        return (hint, iters, liters)
    else:
        solution.assign(solution + du)
        infeasibility = assemble(problem.infeasibility(solution, lb, ub, newmu, params))
        info_green("Tangent linearisation success, infeasibility of guess %.5e" % (infeasibility))
        return (hint, iters, liters)

def secant(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint):

    newmu = float(newmu)
    oldmu = float(oldmu)
    oldsolution = solution.copy(deepcopy = True)
    #dm = problem._dm


    if hint[0] == None:
        pass
    else:
        # Unpack previous solution
        (prevsolution, prevmu) = hint

        du = Function(solution.function_space())
        # Find secant predictor
        multiplier = (newmu-oldmu)/(oldmu-prevmu)
        du.assign((oldsolution - prevsolution)*multiplier)
        infeasibility = assemble(problem.infeasibility(solution, Constant(0), Constant(1), newmu, params))
        info_green("Found secant predictor, infeasibility of guess %.5e" % (infeasibility))
        solution.assign(solution + du)

    hint = [oldsolution, float(oldmu)]

    return (hint, 0, 0)


def nothing(problem, solution, test, trial, oldmu, newmu, k, params, vi, task, hint=None):
    return (hint, 0, 0)
