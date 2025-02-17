__version__ = "2020.0.1"

from .deflation import getEdy, setSnesMonitor, setSnesBounds, compute_tau, DeflatedKSP, SnesDeflator, DeflationOperator, ShiftedDeflation, newton
from .drivers import deflated_barrier
from .gridsequencing import gridsequencing
from .problemclass import PrimalInteriorPoint, PrimalDualInteriorPoint
from .mlogging import info_blue, info_red, info_green
from .nonlinearsolver import SNUFLSolver
from .nonlinearproblem import GeneralProblem
from .prediction import feasibletangent, tangent, secant, nothing
from .misc import plus
from .visolvers import vec, BensonMunson, HintermullerItoKunisch, ProjectedNewton, fb
# from .mg import * # Can crash depending on Ubuntu version


