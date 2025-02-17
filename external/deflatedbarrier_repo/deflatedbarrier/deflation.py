import dolfin
from dolfin import *
from dolfin import PETScOptions, PETScVector, as_backend_type
from ufl import product
from petsc4py import PETSc
from numpy import isnan
from .nonlinearsolver import *
from .nonlinearproblem import *

def getEdy(deflation, y, dy, vi_inact):
    deriv = as_backend_type(deflation.derivative(y)).vec()
    if vi_inact is not None:
        deriv_ = deriv.getSubVector(vi_inact)
    else:
        deriv_ = deriv

    out = -deriv_.dot(dy)

    if vi_inact is not None:
        deriv.restoreSubVector(vi_inact, deriv_)

    return out

def setSnesMonitor(prefix):
    PETScOptions.set(prefix + "snes_monitor_cancel")
    PETScOptions.set(prefix + "snes_monitor")

def setSnesBounds(snes, bounds):
    (lb, ub) = bounds
    snes.setVariableBounds(as_backend_type(lb.vector()).vec(), as_backend_type(ub.vector()).vec())

def compute_tau(deflation, state, update_p, vi_inact):
    if deflation is not None:
        Edy = getEdy(deflation, state, update_p, vi_inact)

        minv = 1.0 / deflation.evaluate(state)
        tau = (1 + minv*Edy/(1 - minv*Edy))
        return tau
    else:
        return 1

class DeflatedKSP(object):
    def __init__(self, deflation, y, ksp, snes):
        self.deflation = deflation
        self.y = y
        self.ksp = ksp
        self.snes = snes
        self.its = 0

    def solve(self, ksp, b, dy_pet):
        # Use the inner ksp to solve the original problem
        self.ksp.setOperators(*ksp.getOperators())
        self.ksp.solve(b, dy_pet)
        self.its += self.ksp.its
        deflation = self.deflation

        if self.snes.getType().startswith("vi"):
            vi_inact = self.snes.getVIInactiveSet()
        else:
            vi_inact = None

        tau = compute_tau(deflation, self.y, dy_pet, vi_inact)
        dy_pet.scale(tau)

        ksp.setConvergedReason(self.ksp.getConvergedReason())

    def reset(self, ksp):
        self.ksp.reset()

    def view(self, ksp, viewer):
        self.ksp.view(viewer)

class SnesDeflator(object):
    def __init__(self, snes, deflation, u):
        self.snes = snes
        self.deflation = deflation
        self.u = u

    def __enter__(self):
        snes = self.snes
        oldksp = snes.ksp
        defksp = DeflatedKSP(self.deflation, self.u, oldksp, snes)
        snes.ksp = PETSc.KSP().createPython(defksp, snes.comm)
        snes.ksp.pc.setType('none')

        self.oldksp = oldksp
        self.defksp = defksp

    def __exit__(self, *args):
        del self.defksp.snes # clean up circular references
        self.snes.ksp = self.oldksp # restore old KSP
        self.snes.setAttr("total_ksp_its", self.defksp.its)

class DeflationOperator(object):
    """
    Base class for deflation operators.
    """
    def set_parameters(self, params):
        self.parameters = params

    def deflate(self, roots):
        self.roots = roots

    def evaluate(self):
        raise NotImplementedError

    def derivative(self):
        raise NotImplementedError

class ShiftedDeflation(DeflationOperator):
    """
    The shifted deflation operator presented in doi:10.1137/140984798.
    """
    def __init__(self, power, shift, normsq=None):
        self.power = power
        self.shift = shift
        self.roots = []
        if normsq is None:
            normsq = lambda u, v, params: inner(u-v, u-v)*dx
        self.normsqc = normsq
        self.parameters = None

    def normsq(self, y, root):
        out = self.normsqc(y, root, self.parameters)
        return out

    def evaluate(self, y):
        m = 1.0
        for root in self.roots:
            normsq = assemble(self.normsq(y, root))
            factor = normsq**(-self.power/2.0) + self.shift
            m *= factor

        return m

    def derivative(self, y):
        if len(self.roots) == 0:
            deta = Function(y.function_space()).vector()
            return deta

        p = self.power
        factors  = []
        dfactors = []
        dnormsqs = []
        normsqs  = []

        for root in self.roots:
            form = self.normsq(y, root)
            normsqs.append(assemble(form))
            dnormsqs.append(assemble(derivative(form, y)))

        for normsq in normsqs:
            factor = normsq**(-p/2.0) + self.shift
            dfactor = (-p/2.0) * normsq**((-p/2.0) - 1.0)

            factors.append(factor)
            dfactors.append(dfactor)

        eta = product(factors)

        deta = Function(y.function_space()).vector()

        for (solution, factor, dfactor, dnormsq) in zip(self.roots, factors, dfactors, dnormsqs):
            deta.axpy(float((eta/factor)*dfactor), dnormsq)

        return deta

def newton(F, J, y, bcs, solverclass, params, solver_params,
           teamno, deflation=None, dm=None, vi=None, prefix=""):

    comm = y.function_space().mesh().mpi_comm()

    problem =  GeneralProblem(F, y, bcs)
    problem.deflation = deflation

    setSnesMonitor(prefix)

    base = SNUFLSolver(problem,solver_parameters=solver_params, dm=dm)
    solver = solverclass(problem, base, params, solver_params, prefix="", vi = vi, dm=dm)
    snes = solver.snes

    oldksp = snes.ksp
    #oldksp.incrementTabLevel(teamno*2)
    defksp = DeflatedKSP(deflation, y, oldksp, snes)
    snes.ksp = PETSc.KSP().createPython(defksp, comm)
    snes.ksp.pc.setType('none')


    try:
        solver.solve()
    except ConvergenceError:
        pass
    except:
        import traceback
        traceback.print_exc()
        pass

    success = snes.getConvergedReason() > 0
    iters   = snes.getIterationNumber()
    liters = snes.getLinearSolveIterations()
    snes.destroy() # this fixes the memory leak!!
    return (success, iters, liters)
