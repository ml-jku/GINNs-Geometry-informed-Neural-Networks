from __future__ import absolute_import

from dolfin import derivative

class GeneralProblem(object):
    def __init__(self, F, y, bcs, J=None, P=None, **kwargs):
        self.F = F
        self.u = y  # Firedrake convention
        self.comm = y.function_space().mesh().mpi_comm()
        self.bcs = bcs
        self.J = J if J is not None else derivative(F, y)
        self.P = P
        for key in kwargs:
            assert key not in ['u', 'comm', 'J', 'P', 'F', 'bcs']
            setattr(self, key, kwargs[key])
