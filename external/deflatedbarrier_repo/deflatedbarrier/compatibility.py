import dolfin
from petsc4py import PETSc

def make_comm(comm):
    if hasattr(dolfin, "has_pybind11") and dolfin.has_pybind11():
        return comm
    elif dolfin.__version__ >= "2018.1.0":
        return comm
    else:
        return PETSc.Comm(comm)

if PETSc.Sys.getVersion()[0:2] <= (3, 7) and PETSc.Sys.getVersionInfo()['release']:

    def get_deep_submat(mat, isrow, iscol=None, submat=None):
        """Get deep submatrix of mat"""
        return mat.getSubMatrix(isrow, iscol, submat=submat)

    def get_shallow_submat(mat, isrow, iscol=None):
        """Get shallow submatrix of mat"""
        submat = PETSc.Mat().create(mat.comm)
        return submat.createSubMatrix(mat, isrow, iscol)

else:

    def get_deep_submat(mat, isrow, iscol=None, submat=None):
        """Get deep submatrix of mat"""
        return mat.createSubMatrix(isrow, iscol, submat=submat)

    def get_shallow_submat(mat, isrow, iscol=None):
        """Get shallow submatrix of mat"""
        submat = PETSc.Mat().create(mat.comm)
        return submat.createSubMatrixVirtual(mat, isrow, iscol)
