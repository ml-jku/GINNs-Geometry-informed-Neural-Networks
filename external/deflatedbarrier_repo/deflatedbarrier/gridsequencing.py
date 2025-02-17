# -*- coding: utf-8 -*-
from dolfin import *
from .deflation import newton
from .compatibility import make_comm
from .drivers import visolver, deflated_barrier
# from .mg import create_dm # Can crash with depending on Ubuntu version
from .mlogging import *
from mpi4py import MPI
import os
import resource
import shutil



def gridsequencing(problem, sharpness_coefficient, branches, params=None, pathfile = None,
                   initialpathfile = None, comm=MPI.COMM_WORLD,
                   mu_start_refine = 0.0, mu_start_continuation = 0.0,
                   iters_total = 20, parameter_update = None, greyness_tol = 1e-1,
                   grid_refinement = 10, parameter_continuation = True):


    dolfin_comm = make_comm(comm)
    mu = Constant(0.0)
    epsilon_original = params[sharpness_coefficient]
    problem = visolver(problem, "BensonMunson")
    if pathfile == None: pathfile = "gs_output"

    for branch in branches:
        params[sharpness_coefficient] = epsilon_original
        mesh = problem.mesh(comm=dolfin_comm)
        gsproblem = GridSequenceProblem(problem, mesh)
        pvd = File(pathfile+"/paraview/refine-branch-%s.pvd"%branch)
        initialstring = pathfile + "/tmp/%s.xml.gz"%(branch)
        tmppathfile = pathfile + "/tmp"
        (F,J,bcs,sp,vi,dm,z,v,w) = requirements(mesh, gsproblem, mu, branch, params)
        if initialpathfile == None:
            h5 = HDF5File(dolfin_comm, "output/mu-%.12e-hmin-%.3e-params-%s-solver-BensonMunson/%s.xml.gz" % (float(mu), mesh.hmin(), params , branch), "r")
        else:
            h5 = HDF5File(dolfin_comm, initialpathfile, "r")
        h5.read(z, "/guess"); del h5

        epsilon = params[sharpness_coefficient]
        # params[sharpness_coefficient] = epsilon
        info_blue(r"Checking solution to initial grid for branch %s, sharpness coefficient = %s"%(branch,epsilon))
        (success,_,_) = newton(F, J, z, bcs, problem.solver,params, sp, None, None, dm, vi)
        if success:
            save_pvd(pvd, z, mu)
        else:
            info_red(r"Solution not found")
            break

        iters = 0
        gr = 1
        while (iters < iters_total):
            # shutil.rmtree(pathfile +"/tmp", ignore_errors=True)
            if gr <= grid_refinement:
                mesh_ = interface_refine(z, mesh, greyness_tol)
                (_,_,_,_,_,_,z_,_,_) = requirements(mesh_, problem, mu, branch, params)
                print('mesh size: %s' %mesh_.hmin())
            else:
                mesh_ = mesh
                z_ = z
                print('mesh size: %s' %mesh_.hmin())

            gsproblem = GridSequenceProblem(problem, mesh_)

            # solve for refined grid before updating epsilon
            if gr <= grid_refinement:
                gr += 1
                info_blue(r"Refining grid for branch %s, sharpness coefficient = %s"%(branch,epsilon))
                prolong(z,z_)
                exists = os.path.isfile(tmppathfile +"/%s.xml.gz"%branch)
                if exists: os.remove(tmppathfile +"/%s.xml.gz"%branch)
                problem.save_solution(mesh_,z_,mu,params,0,branch,tmppathfile)
                ([z_],_) = deflated_barrier(gsproblem, params, mu_start=mu_start_refine, mu_end = 1e-10, max_halfstep = 1, initialstring = initialstring)
                save_pvd(pvd, z_, mu)
            # newton(F, J, z_, bcs, params, sp, None, None, None, vi)

            if parameter_continuation == True:
                if parameter_update == None:
                    raise("Require rules for update in continuation parameter")
                epsilon = parameter_update(epsilon, z_)
                params[sharpness_coefficient] = epsilon
                info_blue(r"Solve for new sharpness coefficient = %s for branch %s"%(epsilon, branch))
                exists = os.path.isfile(tmppathfile +"/%s.xml.gz"%branch)
                if exists: os.remove(tmppathfile +"/%s.xml.gz"%branch)
                problem.save_solution(mesh_,z_,mu,params,0,branch,tmppathfile)
                ([z_],_) = deflated_barrier(gsproblem, params, mu_start=mu_start_continuation, mu_end = 1e-10, max_halfstep = 0, initialstring = initialstring)

            z = z_
            mesh = mesh_
            save_pvd(pvd, z, mu)
            problem.save_solution(mesh,z,mu,params,0,branch,pathfile+"/cont-%s/branch-%s/iter-%s"%(epsilon,branch,iters))
            File(comm,pathfile+"/cont-%s/branch-%s/iter-%s/mesh.xml"%(epsilon, branch,iters)) << mesh
            iters += 1
        # shutil.rmtree("gs_output/tmp", ignore_errors=True)
        info_blue(r"Reached target refinement, terminating algorithm")

    shutil.rmtree(tmppathfile, ignore_errors=True)
    return None

def requirements(mesh, problem, mu, branch, params):
    Z = problem.function_space(mesh)
    # dm = create_dm(Z, problem)
    dm = None
    z = Function(Z, name = "Solution")
    v = TestFunction(Z)
    w = TestFunction(Z)
    (lb, ub)  = problem.bounds(mesh, mu, params)
    vi = problem.bounds_vi(Z, mu, params)
    sp = problem.solver_parameters(float(mu), branch, "ContinuationTask", params)

    F = problem.residual(z, v, lb, ub, mu, params)
    J = problem.jacobian(F, z, params, v, w)
    bcs = problem.boundary_conditions(Z, params)
    return(F,J,bcs,sp,vi,dm,z,v,w)

def interface_refine(z, mesh, greyness_tol):
    rho = z.split(True)[0]

    for i in range(1):
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        for cell in cells(mesh):
            p = cell.midpoint()
            if greyness_tol < rho(p) < 1. - greyness_tol:
                cell_markers[cell] = True
        mesh = refine(mesh, cell_markers)

    return mesh

def save_pvd(pvd, z, mu):
    rho_ = z.split(deepcopy=True)[0]
    rho_.rename("Control", "Control")
    pvd << rho_


def prolong(z,z_):
    subz  = z.split(deepcopy=True)
    subz_ = z_.split(deepcopy=True)

    for (u, u_) in zip(subz, subz_):
        ele = u.function_space().ufl_element()
        if ele.family() == "Real":
            u_.vector().set_local(u.vector().get_local())
        else:
            mat = PETScDMCollection.create_transfer_matrix(u.function_space(), u_.function_space())
            mat.mult(u.vector(), u_.vector())
            as_backend_type(u_.vector()).update_ghost_values()

    for (i, u_) in enumerate(subz_):
        assign(z_.sub(i), u_)

class GridSequenceProblem(object):
    def __init__(self, problem, mesh_):
        self.mesh_ = mesh_
        self.problem = problem
    def mesh(self, comm):
        return self.mesh_
    def save_solution(self, mesh, z, mu, params, iter_subproblem, branch, pathfile):
        pass
    def save_pvd(self, pvd, u, mu):
        pass
    def number_solutions(self, mu, params):
        return 1
    def __getattr__(self, attr):
        return getattr(self.problem, attr)
