from dolfin import *
import sys
import os
from .mlogging import *
import numpy as np

def soln(array):
    if len(array) == 1:
        return "solution"
    else:
        return "solutions"

def plus(x):
    return conditional(gt(x, 0), x, 0)

def MorYos(x):
    return conditional(gt(x, 0), 0, x**2)

def create_output_folder(saving_folder):
    if sys.version_info[0] < 3:
        try:
            os.removedirs(saving_folder + "output")
        except:
            pass
        try:
            os.makedirs(saving_folder + "output")
        except:
            pass
    else:
        os.makedirs(saving_folder + "output", exist_ok=True)
    return None

def inertia_switch(inertia, inertia_old, max_solutions, Max_solutions, branches, branch):
    try: # interia_old[branch] may not be defined yet
        if inertia[branch] != inertia_old[branch]:
            info_green("Inertia change detected")
            if max_solutions < Max_solutions: # do not want to exceed known max solution count
                max_solutions +=1
                for iter in range(Max_solutions):
                    if found_solutions[iter] == 0:
                        branches.append(iter)
                        break

    except: pass
    return [max_solutions, branches]

def report_profile(Log, Max_solutions):

    out = []
    print("-" * 80)
    print("| Profiling statistics collected" + " "*35 + "|")
    print("-" * 80)

    print(" " + "*"*21)

    for branch in range(Max_solutions):
        cont = np.asarray(Log["num_snes_its"][branch])
        cont = np.sum(cont)
        defl = np.asarray(Log["num_snes_its_defl"][branch])
        defl = np.sum(defl)
        pred = np.asarray(Log["num_snes_its_pred"][branch])
        pred = np.sum(pred)
        cost = Log["solutions_cost"][branch][-1]
        min  = Log["min"][branch]
        print(" * Branch %s *"%branch)
        print(" " + "*"*21)
        print("     Cost:                         %s" %cost)
        print("     Local minimum:                %s" %min)
        print("     Continuation iterations:      %s" %cont)
        print("     Deflation iterations:         %s" %defl)
        print("     Prediction iterations:        %s" %pred)
        print()
        out.append("%d"%cont)
        out.append("%d"%defl)
        out.append("%d"%pred)

    out = np.array([out])
    return out
