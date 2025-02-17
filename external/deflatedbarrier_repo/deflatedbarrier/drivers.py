# -*- coding: utf-8 -*-
from dolfin import *
from .mlogging import *
from .deflation import newton
from .compatibility import make_comm
from .misc import create_output_folder, inertia_switch, report_profile, MorYos
from .visolvers import BensonMunson, HintermullerItoKunisch, ProjectedNewton
# from .mg import create_dm # Can crash depending on Ubuntu version
from mpi4py import MPI
from copy import deepcopy
import os
import resource
import shutil

def deflated_barrier(problem, params=None, solver = "BensonMunson",
                  comm=MPI.COMM_WORLD, mu_start=1000,
                  mu_end = 1e-15, hint=None,
                  max_halfstep = 1, initialstring=None,
                  check_volume_constraint = True, saving_folder = ""):
    # inelegant ways of trying to make the output folder...
    create_output_folder(saving_folder)
    #  Overload problem classes e.g. if we wish to use Hintermuller-Ito-Kunisch primal-dual active set strategy
    problem = visolver(problem, solver)

    iter_subproblem = 0 # start iteration count

    # Setup mesh and other FEM requirements
    (mu, dolfin_comm, mesh, FcnSpace, dm) = FEMsetup(comm, problem, mu_start)
    problem._dm = dm

    guesses = extractguesses(problem, initialstring, FcnSpace, params, dolfin_comm)
    number_initial_branches = len(guesses)


    Max_solutions = problem.number_solutions(0, params)
    found_solutions = [1]*number_initial_branches
    hmin = mesh.hmin()
    #FIXME smart way of knowing max number of solutions?
    [guesses, found_solutions] = initialise_guess(guesses, Max_solutions, found_solutions, FcnSpace, params)

    hint = [[None, 0.0]]*Max_solutions
    hint_guess = [[None, 0.0]]*Max_solutions

    oldguesses = [guess.copy(deepcopy=True) for guess in guesses]
    deflation = problem.deflation_operator(params)

    u = Function(FcnSpace)
    v = TestFunction(FcnSpace)
    w = TrialFunction(FcnSpace)
    bcs = problem.boundary_conditions(FcnSpace, params)

    oldmu = Constant(0)
    halfmu = Constant(0)
    num_halfstep = 0
    branch_deflate_start = 0
    inertia = {}
    inertia_old = {}
    # the bounds that are passed to the solver
    vi = problem.bounds_vi(FcnSpace, mu, params)

    # Book-keeping
    Log = createLog(Max_solutions)

    # start loop!
    while True:
        # Useful if looking for mememory leaks
        # info_red("Memory used: %s" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        # If there are no guesses or active branches, then algorithm has failed
        if sum(found_solutions) == 0:
            info_red("All branches have failed, algorithm stopping.")
            break

        info_blue("Considering mu = %s" % float(mu))
        deflation.deflate([])

        # the bounds used in the barrier method
        (lb, ub)  = problem.bounds(mesh, mu, params)

        # FIXME why does this have to be within the while loop?
        F = problem.residual(u, v, lb, ub, mu, params)
        J = problem.jacobian(F,u,params,v,w)
        solutions = []

        # increment iteration index
        iter_subproblem += 1

        # the branch to be deflated from in DeflationTask
        branch_deflate = branch_deflate_start

        solutionpath_string = (saving_folder + "output/mu-%.12e-hmin-%.3e-params-%s-solver-%s" %(float(mu),hmin,params,solver))
        pvd = File("%s/solutions.pvd" %solutionpath_string)

        max_solutions = problem.number_solutions(mu, params)
        # which branches to run through
        branch_iter = 0
        branches = initialise_branches(found_solutions, Max_solutions, max_solutions)
        max_solutions = max(max_solutions, sum(found_solutions))


        while branch_iter < len(branches):
             outputarg = PredictionCorrectionDeflation(iter_subproblem, problem, FcnSpace, mesh,
                                                      u, v, w, lb, ub, params, F, J, bcs, dm, deflation,
                                                      dolfin_comm,
                                                      branches, branch_iter, solutionpath_string,
                                                      branch_deflate_start, branch_deflate,
                                                      hint_guess, hint, guesses,
                                                      mu, oldmu, oldguesses, found_solutions, solutions,
                                                      pvd, inertia, inertia_old,
                                                      max_solutions, Max_solutions, halfmu, num_halfstep, max_halfstep, Log,
                                                      check_volume_constraint)


             (branch_iter, branch_deflate, mu, oldmu, oldguesses, Log, found_solutions,
             deflation, halfmu, num_halfstep, max_solutions, guesses, task) = outputarg



        for branch in range(Max_solutions):
            if found_solutions[branch] == 1:
                Log["solutions_cost"][branch].append(problem.cost(guesses[branch], params))
            else:
                Log["solutions_cost"][branch].append("NaN")

        if float(mu) == 0.0:
            info_blue("Terminating because we have reached target mu")
            break
        if float(mu) <= float(mu_end):
            mu.assign(Constant(0.0))

        newparams = UpdateSubproblemParams(oldguesses, guesses, Max_solutions,
                                task, num_halfstep, max_halfstep,
                                hint_guess, hint,
                                FcnSpace, params, u, problem,
                                oldmu, mu, halfmu, iter_subproblem, Log)

        (oldguesses, hint, oldmu, mu, num_halfstep) = newparams

        # Live simplistic tracking of data
        Log["mus"].append(float(mu))
        save = open(saving_folder + "output/DABlog.txt","w")
        for i in range(Max_solutions):
            save.write( "%s,"%i+str(Log["solutions_cost"][i])[1:-1] +"\n" )
        # save.write( str(Log["num_snes_its"]) )
        # save.write( str(Log["num_snes_its_defl"]) )
        # save.write( str(Log["num_snes_its_pred"]) )
        save.write( "NaN," + str(Log["mus"])[1:-1] )
        save.close()

    out = report_profile(Log, Max_solutions)
    return (guesses, out)

def initialise_guess(guesses,Max_solutions,found_solutions, V, params):
    number_initial_branches = len(guesses)
    if (Max_solutions - number_initial_branches) > 0:
        for i in range(number_initial_branches,Max_solutions):
            guesses.append(Function(V))
            guesses[i].assign(guesses[0])
            found_solutions.append(0)
    return [guesses, found_solutions]

def visolver(problem, solver):
    if solver == "BensonMunson":
        problem = BensonMunson(problem)
    elif solver == "HintermullerItoKunisch":
        problem = HintermullerItoKunisch(problem)
    elif solver == "ProjectedNewton":
        problem = ProjectedNewton(problem)
    return problem

def FEMsetup(comm, problem, mu_start):
    mu = Constant(mu_start)
    dolfin_comm = make_comm(comm)
    mesh = problem.mesh(comm=dolfin_comm)
    FcnSpace = problem.function_space(mesh)
    # dm = create_dm(FcnSpace, problem)
    dm = None
    return (mu, dolfin_comm, mesh, FcnSpace, dm)

def extractguesses(problem, initialstring, FcnSpace, params, dolfin_comm):
    if initialstring != None:
        guesses = [Function(FcnSpace)]
        h5 = HDF5File(dolfin_comm, initialstring, "r")
        h5.read(guesses[0], "/guess")
        del h5
    else:
        guesses = problem.initial_guesses(FcnSpace, params)
    return guesses

def createLog(Max_solutions):
    Log = {}
    Log["num_ksp_its"]      = [0]*Max_solutions
    Log["num_ksp_its_pred"] = [0]*Max_solutions
    Log["num_ksp_its_defl"] = [0]*Max_solutions
    Log["mus"]              = []
    Log["solutions_cost"]   = {}
    Log["num_snes_its"]     = {}
    Log["num_snes_its_defl"]= {}
    Log["num_snes_its_pred"]= {}
    Log["min"]= {}
    for i in range(Max_solutions):
        Log["solutions_cost"][i]    = []
        Log["num_snes_its"][i]      = []
        Log["num_snes_its_defl"][i] = []
        Log["num_snes_its_pred"][i] = []
        Log["min"][i] = "No"
    return Log

def initialise_branches(found_solutions, Max_solutions, max_solutions):
    branches = []
    iter = 0
    # Run through active solution branches
    for iter in range(Max_solutions):
        if found_solutions[iter] == 1:
            branches.append(iter)
    iter = 0
    iter2 = 0
    # Also initialise branches that the user has specified exist
    if max_solutions > sum(found_solutions):
        while iter2 < max_solutions - sum(found_solutions):
            if found_solutions[iter] == 0:
                branches.append(iter)
                iter2+=1
            iter +=1
    return branches

def PredictionCorrectionDeflation(iter_subproblem, problem, FcnSpace, mesh,
                                         u, v, w, lb, ub, params, F, J, bcs, dm, deflation,
                                         dolfin_comm,
                                         branches, branch_iter, solutionpath_string,
                                         branch_deflate_start, branch_deflate,
                                         hint_guess, hint, guesses,
                                         mu, oldmu, oldguesses, found_solutions, solutions,
                                         pvd, inertia, inertia_old,
                                         max_solutions, Max_solutions, halfmu, num_halfstep,max_halfstep, Log,
                                         check_volume_constraint):

    def outputargs():
        return (branch_iter, branch_deflate, mu, oldmu, oldguesses, Log, found_solutions,
                deflation, halfmu, num_halfstep, max_solutions, guesses, task)

    branch = branches[branch_iter]

    # the bounds that are passed to the solver
    vi = problem.bounds_vi(FcnSpace, mu, params)

    # If solution is already saved, no need to recalculate it
    exists = os.path.isfile("%s/%s.xml.gz" % (solutionpath_string, branch))
    if exists:
        hint_tmp = u.copy(deepcopy = True)
        h5 = HDF5File(dolfin_comm, "%s/%s.xml.gz" % (solutionpath_string, branch), "r")
        h5.read(u, "/guess")
        del h5
        info_green("Solution already found")
        hint_guess[branch][0] = hint_tmp
        hint_guess[branch][1] = float(oldmu)
        task = "ContinuationTask"
    else:
        # If branch is active, continue the branch
        if found_solutions[branch] == 1:
            u.assign(oldguesses[branch])
            task = "PredictorTask"
            info_blue("Task: %s, Branch: %s" %(task, branch))
            # solver parameters that are passed to the solver
            sp = problem.solver_parameters(mu, branch, task, params)
            # Predictor-corrector scheme
            if float(mu) != 0.0:
                (hint_guess[branch], its0, lits0) = problem.predictor(problem, u, v, w,
                                                                      oldmu, mu, iter_subproblem,
                                                                      params, task, vi, hint[branch])

                Log["num_ksp_its_pred"][branch]  += lits0
                Log["num_snes_its_pred"][branch].append(its0)
            task = "ContinuationTask"

        # If branch is inactive, perform deflation
        elif found_solutions[branch] == 0:
            task = "DeflationTask"

            if branch == 0 and branch_deflate == 0: branch_deflate = branches[0]
            u.assign(oldguesses[branch_deflate])

            # elif task == "InfeasibleRhoTask":
            #     u.assign(oldguesses[branch])

    if task == "ContinuationTask":
        info_blue("Task: %s, Branch: %s, mu: %s" %(task, branch, float(mu)))
    elif task == "DeflationTask":
        info_blue("Task: %s, Branch: %s, Initial guess: branch %s, mu: %s" %(task, branch, branch_deflate, float(mu)))

    # solver parameters that are passed to the solver
    sp = problem.solver_parameters(mu, branch, task, params)

    (success, its, lits) = newton(F, J, u, bcs,
                     problem.solver,
                     params,
                     sp,
                     None, deflation, dm, vi)

    # Hopefully, scheme has converged
    if success:
        if task == "ContinuationTask":
            # count iterations
            Log["num_snes_its"][branch].append(its)
            Log["num_ksp_its"][branch]  += lits
        elif task == "DeflationTask":
            Log["num_snes_its_defl"][branch].append(its)
            Log["num_ksp_its_defl"][branch]  += lits

        # sometimes found solution violates rho volume constraint and we should
        # not continue these solutions.
        rho = split(u)[0]
        infeasibility_rho = assemble(rho*dx)/assemble(Constant(1.0)*dx(mesh))

        if check_volume_constraint and (infeasibility_rho > problem.volume_constraint(params)+1e-4):
            found_solutions[branch] = 0

            lmbda = split(u)[-1]
            infeasibility_lmbda = assemble(lmbda*dx)/assemble(Constant(1.0)*dx(mesh))
            info_red(r"Found solution violates volume constraint on rho, rho*dx/$|\Omega|$: %s,lmbda %s\nDeflating non-feasible solution and trying again" %(infeasibility_rho, infeasibility_lmbda))
            task = "InfeasibleRhoTask"

        else:
            found_solutions[branch] = 1 # keep track of successful branch continuation

            guesses[branch].assign(u)

            problem.save_pvd(pvd, u, mu)
            problem.save_solution(mesh, u, mu, params, iter_subproblem, branch, solutionpath_string)

            if float(mu) == 0.0:
                rho = split(u)[0]
                lb_inertia = Constant(0)
                ub_inertia = Constant(1)
                L = ( problem.lagrangian(u, params)
                      + 1e10*MorYos(rho - lb_inertia)*dx
                      + 1e10*MorYos(ub_inertia - rho)*dx
                    )
                F = derivative(L, u, v)
                J = problem.jacobian(F,u,params,v,w)
                inertia[branch] = problem.compute_stability(mu, params, lb, ub, branch, u, v, w, FcnSpace, bcs, J)
                if problem.expected_inertia() == None:
                    Log["min"][branch] = "Unknown"
                elif inertia[branch][0] == problem.expected_inertia():
                    Log["min"][branch] = "Yes"

            info_green(r"Found solution in branch %d for mu = %s" % (branch, float(mu)))
            branch_iter +=1


        solutions.append(u.copy(deepcopy=True))
        deflation.deflate(solutions)
        branch_deflate = branch_deflate_start



        if len(solutions) >= max(max_solutions, sum(found_solutions)) and task != 'InfeasibleRhoTask':
            info_red("Not deflating as we have found the maximum number of solutions for given mu")
            package = outputargs()
            return package
    else:
        if task == "ContinuationTask":
            # keep track of failed iterations too
            Log["num_snes_its"][branch].append(its)
            # if max_halfstep == True and num_halfstep == 0:
            if num_halfstep < max_halfstep:
                # if continuation has failed, half mu stepsize
                halfmu.assign(mu)
                mu.assign(0.5*(float(mu)+float(oldmu)))
                info_red("%s for branch %s has failed, halfing stepsize in mu, considering mu = %s" %(task,branch, float(mu)))
                branch_deflate = 0
                num_halfstep += 1
                branch_iter = Max_solutions # to break the while loop
                task = "HalfstepsizeTask"
                package = outputargs()
                return package
            else:
                # already attempted to half stepsize in mu, time to move on...
                found_solutions[branch] = 0
                branch_iter +=1
                num_halfstep = 0


        elif task == 'DeflationTask':
            # if deflation failed, use a different branch as an initial guess

            branch_deflate +=1
            found_solutions[branch] = 0
            try:
                if branch_deflate == branch: branch_deflate +=1 # should not use one's own branch as an initial guess in deflation
                while found_solutions[branch_deflate] == 0: branch_deflate +=1 # should not use branch with no current solution
            except:
                pass
            if branch_deflate > max_solutions-1: # if deflation has failed from all branches, move on...
                info_red("No solution found for branch %s, moving onto next branch" %branch)
                while found_solutions[branch] == 0:
                    branch_iter += 1
                    if branch_iter==max_solutions:
                        package = outputargs()
                        return package
                branch_deflate = branch_deflate_start
        if sum(found_solutions) == 0:
            package = outputargs()
            return package

    return outputargs()

def UpdateSubproblemParams(oldguesses, guesses, Max_solutions,
                        task, num_halfstep, max_halfstep,
                        hint_guess, hint,
                        FcnSpace, params, u, problem,
                        oldmu, mu, halfmu, iter_subproblem, Log):

    def outputargs():
        return (oldguesses, hint, oldmu, mu, num_halfstep)

    # This is complicated by the half step procedure. Need to ensure to correct
    # hints and oldmu is passed on if half steps are being used

    # update the oldguesses
    for i in range(Max_solutions):
        oldguesses[i].assign(guesses[i])
    if task == "HalfstepsizeTask":
        pass # if in halfstep mode, do nothing

    # if half step has suceeded, then update hints for predictor task and let the
    # new mu be the previously failed mu.
    elif num_halfstep <= max_halfstep and num_halfstep !=0:
        # If hints are empty, can skip this
        for branch in range(Max_solutions):
            if hint_guess[branch][0] is not None:
                # If hint is empty, then it needs to be initialised
                if hint[branch][0] == None:
                    hint[branch] = [Function(FcnSpace), 0.0]
                hint[branch][0].assign(hint_guess[branch][0])
                hint[branch][1] = deepcopy(hint_guess[branch][1])
        oldmu.assign(mu)
        mu.assign(halfmu)
        num_halfstep = 0
    # If there was no half step happening, then proceed as normal. Update hints
    # and update mu accordingly
    else:
        for branch in range(Max_solutions):
            if hint_guess[branch][0] is not None:
                if hint[branch][0] == None:
                    hint[branch] = [Function(FcnSpace), 0.0]
                hint[branch][0].assign(hint_guess[branch][0])
                hint[branch][1] = deepcopy(hint_guess[branch][1])
        k_mu_old = float(mu)/float(oldmu) if (iter_subproblem>1 and float(oldmu) != 0.) else "NaN"
        oldmu.assign(mu)
        if float(mu) > 0.0:
            mu.assign(problem.update_mu(u, float(mu), min(Log["num_snes_its"]), iter_subproblem, k_mu_old, params))
    return outputargs()
