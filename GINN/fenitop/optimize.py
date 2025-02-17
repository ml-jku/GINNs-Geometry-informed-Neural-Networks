"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

import numpy as np
from mpi4py import MPI
from scipy import sparse as sparse
from scipy.linalg import solve


def optimality_criteria(rho, rho_min, rho_max, V, dCdrho, dVdrho, move=0.05):
    """Solution update scheme with optimality criteria (OC)."""
    lb, ub = 0.0, 1e6
    comm = MPI.COMM_WORLD
    while ub-lb > 1e-4:
        mid = (lb+ub) / 2.0
        rho_new = np.maximum.reduce([np.minimum.reduce(
            [rho*(-dCdrho/(dVdrho+1e-12)/mid)**0.5, rho+move, rho_max]), rho-move, rho_min])
        dV = comm.allreduce(dVdrho@(rho_new-rho), op=MPI.SUM)
        if V + dV > 0:
            lb = mid
        else:
            ub = mid
    change = comm.allreduce(np.max(np.abs(rho_new-rho), initial=0), op=MPI.MAX)
    return rho_new, change


def mma_optimizer(m, n, opt_iter, xval, xmin, xmax, xold1, xold2, df0dx, fval,
                  dfdx, low, upp, a0=1, a=None, c=None, d=None, move=0.05,
                  asyinit=0.5, asydecr=0.7, asyincr=1.2,
                  low_bnd=0.002, up_bnd=1.0, albefa=0.1, feps=1e-6):
    """Solution update scheme with the method of moving asymptotes (MMA).
    The algorithm is available in https://doi.org/10.1002/nme.1620240207.

    Minimize:
        f_0(x) + a_0*z + sum(c_i*y_i + 0.5*d_i*(y_i)^2)
    Subjected to:
        f_i(x) - a_i*z - y_i <= 0,    i = 1, 2, ..., m
        xmin_j <= x_j <= xmax_j,      j = 1, 2, ..., n
        y_i >= 0,                     i = 1, 2, ..., m
        z >= 0

    Args:
        m: The number of general constraints.
        n: The number of the variables, x_j.
        opt_iter: Iteration counter.
        xval: Current values of the variables, x_j.
        xmin, xmax: Lower and upper bounds of the variables, x_j.
        xold1, xold2: The values of x_j at one and two iterations ago.
        df0dx: The derivatives of the objective function, f_0(x),
            with respect to the variables, x_j, calculated at xval.
        fval: The values of the constraint functions, f_i(x), calculated at xval.
        dfdx: An (m, n) array with the derivatives of the constraint functions,
            f_i(x), with respect to the variables, x_j, calculated at xval.
        low, upp: Lower and upper asymptotes from the previous iteration.
        a0, a, c, d: Coefficients in the objective function.
        move: Move limit of the variables, x_j.
        asyinit: Initial rate of the asymptotes.
        asydecr: Decreasing rate of the asymptotes when the variables are oscillating.
        asyincr: Increasing rate of the asymptotes when the variables are monotonically updated.
        low_bnd, up_bnd: Lower and upper bounds for determining the asymptotes.
        albefa: A parameter for determining the bounds of the variables.
        feps: A parameter for approximating the objective function.

    Returns:
        x_new: Optimal values of the variables, x_j, in the current subproblem.
        change: Maximum change of the variables, x_j.
        low, upp: Lower and upper asymptotes calculated and used in the current subproblem.
    """

    # Initialize the coefficients of the objective function
    if a is None:
        a = np.zeros(m)
    if c is None:
        c = np.full(m, 1000)
    if d is None:
        d = np.zeros(m)
    comm = MPI.COMM_WORLD
    n_global = comm.allreduce(n, op=MPI.SUM)
    epsimin = np.sqrt(m+n_global)*1e-9

    # Evaluate the asymptotes (low and upp)
    xrange = xmax - xmin
    if opt_iter < 2.5:
        low = xval - asyinit*xrange
        upp = xval + asyinit*xrange
    else:
        idx = (xval-xold1) * (xold1-xold2)  # Check the oscillation
        gamma = np.ones(n)
        gamma[idx > 0] = asyincr  # Relax the asymptotes if monotonic
        gamma[idx < 0] = asydecr  # Tighten the asymptotes if oscillating
        low = xval - gamma*(xold1-low)
        upp = xval + gamma*(upp-xold1)
        lowmin = xval - up_bnd*xrange
        lowmax = xval - low_bnd*xrange
        uppmin = xval + low_bnd*xrange
        uppmax = xval + up_bnd*xrange
        low = np.minimum(np.maximum(low, lowmin), lowmax)
        upp = np.maximum(np.minimum(upp, uppmax), uppmin)

    # Evaluate the bounds of the variables (alpha and beta)
    alpha = np.maximum.reduce([xmin, low+albefa*(xval-low), xval-move*xrange])
    beta = np.minimum.reduce([xmax, upp-albefa*(upp-xval), xval+move*xrange])

    # Evaluate the coefficients of the approximating objective fuction (p0 and q0)
    idx = df0dx > 0
    p0, q0 = np.zeros(n), np.zeros(n)
    p0[idx], p0[~idx] = 1.001*df0dx[idx], -0.001*df0dx[~idx]
    q0[idx], q0[~idx] = 0.001*df0dx[idx], -1.001*df0dx[~idx]
    ul1 = upp - low
    p0, q0 = (p0+feps/ul1)*(upp-xval)**2, (q0+feps/ul1)*(xval-low)**2

    # Evaluate the coefficients of the approximating constraint fuctions (P_mat and Q_mat)
    P_mat, Q_mat = np.zeros((m, n)), np.zeros((m, n))
    dfdx = dfdx.reshape(m, n)
    idx = dfdx > 0
    P_mat[idx], Q_mat[~idx] = dfdx[idx], -dfdx[~idx]
    P_mat, Q_mat = P_mat*(upp-xval)**2, Q_mat*(xval-low)**2

    # Evaluate the b vector for constraint functions
    b = P_mat@(1.0/(upp-xval)) + Q_mat@(1.0/(xval-low))
    b = comm.allreduce(b, op=MPI.SUM) - fval

    # Solve the subproblem with a primal-dual interior-point approach
    x_new = solve_subproblem(m, epsimin, low, upp, alpha, beta, p0, q0,
                             P_mat, Q_mat, a0, a, b, c, d)
    change = comm.allreduce(np.max(np.abs(x_new-xval), initial=0), op=MPI.MAX)
    return x_new, change, low, upp


def solve_subproblem(m, epsimin, low, upp, alpha, beta, p0, q0, P_mat, Q_mat,
                     a0, a, b, c, d):
    """Solve the MMA subproblem with a primal-dual interior-point approach.

    Minimize:
        sum[p0j/(uppj-xj) + q0j/(xj-lowj)] + a0*z + sum(ci*yi + 0.5*di*yi^2)
    Subjected to:
        sum[pij/(uppj-xj) + qij/(xj-lowj)] - ai*z - yi <= bi,    i = 1, 2, ..., m
        alphaj <= xj <= betaj,    j = 1, 2, ..., n
        yi >= 0,                  i = 1, 2, ..., m
        z >= 0
    """
    # Set the initial guess for variables and Lagrange multipliers
    comm = MPI.COMM_WORLD
    eps = 1.0
    x = 0.5*(alpha+beta)
    xi = np.maximum(1.0/(x-alpha), 1.0)
    eta = np.maximum(1.0/(beta-x), 1.0)
    _lambda = np.ones(m)
    if comm.rank == 0:
        y, z, s = np.ones(m), 1.0, np.ones(m)
        mu = np.maximum(0.5*c, 1.0)
        zeta = 1.0

    while eps > epsimin:  # Primal-dual interior point method
        p_lambda = p0 + _lambda@P_mat
        q_lambda = q0 + _lambda@Q_mat
        g_vec = P_mat@(1.0/(upp-x)) + Q_mat@(1.0/(x-low))
        g_vec = comm.allreduce(g_vec, op=MPI.SUM)
        dpsi_dx = p_lambda/(upp-x)**2 - q_lambda/(x-low)**2

        # Evaluate the residual norm
        res_x = comm.gather(dpsi_dx-xi+eta, root=0)
        res_xi = comm.gather(xi*(x-alpha)-eps, root=0)
        res_eta = comm.gather(eta*(beta-x)-eps, root=0)
        if comm.rank == 0:
            res_x, res_xi, res_eta = np.hstack(res_x), np.hstack(res_xi), np.hstack(res_eta)
            res_y = c + d*y - mu - _lambda
            res_z = a0 - zeta - a@_lambda
            res_lambda = g_vec - a*z - y + s - b
            res_mu = mu*y - eps
            res_zeta = zeta*z - eps
            res_s = _lambda*s - eps
            res = np.hstack([res_x, res_y, res_z, res_lambda, res_xi, res_eta,
                            res_mu, res_zeta, res_s])
            res_norm = np.linalg.norm(res, ord=2)
            res_max = np.linalg.norm(res, ord=np.inf)
        else:
            res_norm, res_max = None, None
        res_norm = comm.bcast(res_norm, root=0)
        res_max = comm.bcast(res_max, root=0)

        newton_iter = 0
        while res_max > 0.9*eps and newton_iter < 100:  # Newton's method for the search direction
            newton_iter += 1
            if newton_iter == 100 and comm.rank == 0:
                print("Maximum inner iterations reached in the MMA optimizer", flush=True)

            # Evaluate the left-hand side
            p_lambda = p0 + _lambda@P_mat
            q_lambda = q0 + _lambda@Q_mat
            G_mat = P_mat/(upp-x)**2 - Q_mat/(x-low)**2
            G_mat_global = comm.gather(G_mat, root=0)
            diag_x = 2*(p_lambda/(upp-x)**3+q_lambda/(x-low)**3) + xi/(x-alpha) + eta/(beta-x)
            diag_x_global = comm.gather(diag_x, root=0)
            if comm.rank == 0:
                G_mat_global = np.hstack(G_mat_global)
                diag_x_global = np.hstack(diag_x_global)
                diag_y = d + mu/y
                A_mat = sparse.spdiags(s/_lambda+1.0/diag_y, 0, m, m) \
                    + G_mat_global*(1.0/diag_x_global)@G_mat_global.T
                lhs = np.vstack([np.hstack([A_mat, a.reshape(-1, 1)]),
                                 np.hstack([a, -zeta/z])])

            # Evaluate the right-hand side
            dpsi_dx = p_lambda/(upp-x)**2 - q_lambda/(x-low)**2
            delta_x = dpsi_dx - eps/(x-alpha) + eps/(beta-x)
            delta_x_global = comm.gather(delta_x, root=0)
            g_vec = P_mat@(1.0/(upp-x)) + Q_mat@(1.0/(x-low))
            g_vec = comm.allreduce(g_vec, op=MPI.SUM)
            if comm.rank == 0:
                delta_x_global = np.hstack(delta_x_global)
                delta_y = c + d*y - _lambda - eps/y
                delta_z = a0 - a@_lambda - eps/z
                delta_lambda = g_vec - a*z - y - b + eps/_lambda
                b_lambda = delta_lambda + delta_y/diag_y - G_mat_global@(delta_x_global/diag_x_global)
                rhs = np.hstack([b_lambda, delta_z])

            # Solve the search direction
            if comm.rank == 0:
                solution = solve(lhs, rhs)
                incr_lambda, incr_z = solution[:m], solution[m]
                incr_y = -delta_y/diag_y + incr_lambda/diag_y
                incr_mu = -mu + eps/y - (mu*incr_y)/y
                incr_zeta = -zeta + eps/z - zeta*incr_z/z
                incr_s = -s + eps/_lambda - (s*incr_lambda)/_lambda
            else:
                incr_lambda = None
            incr_lambda = comm.bcast(incr_lambda, root=0)
            incr_x = -delta_x/diag_x - (incr_lambda@G_mat)/diag_x
            incr_xi = -xi + eps/(x-alpha) - (xi*incr_x)/(x-alpha)
            incr_eta = -eta + eps/(beta-x) + (eta*incr_x)/(beta-x)
            incr_xi_global = comm.gather(incr_xi, root=0)
            incr_eta_global = comm.gather(incr_eta, root=0)
            xi_global = comm.gather(xi, root=0)
            eta_global = comm.gather(eta, root=0)
            if comm.rank == 0:
                xi_global = np.hstack(xi_global)
                eta_global = np.hstack(eta_global)
                values = np.hstack([y, z, _lambda, xi_global, eta_global, mu, zeta, s])
                incr_xi_global = np.hstack(incr_xi_global)
                incr_eta_global = np.hstack(incr_eta_global)
                incr = np.hstack([incr_y, incr_z, incr_lambda, incr_xi_global,
                                  incr_eta_global, incr_mu, incr_zeta, incr_s])

            # Evaluate the upper bound of the step size
            step_inv_alpha = np.max(-1.01*incr_x/(x-alpha), initial=0)  # Ensure x-alpha >= 0.01/1.01*(x_old-alpha)
            step_inv_beta = np.max(1.01*incr_x/(beta-x), initial=0)  # Ensure beta-x >= 0.01/1.01*(beta-x_old)
            step_inv_alpha = comm.allreduce(step_inv_alpha, op=MPI.MAX)
            step_inv_beta = comm.allreduce(step_inv_beta, op=MPI.MAX)
            if comm.rank == 0:
                step_inv = max(-1.01*incr/values)  # Ensure y >= 0.01/1.01*y_old, etc.
                step_inv = max(step_inv_alpha, step_inv_beta, step_inv, 1.0)  # Ensure step <= 1
                step = 1.0 / step_inv
            else:
                step = None
            step = comm.bcast(step, root=0)

            x_old, lambda_old = x.copy(), _lambda.copy()
            xi_old, eta_old = xi.copy(), eta.copy()
            if comm.rank == 0:
                y_old, z_old = y.copy(), z
                mu_old, zeta_old, s_old = mu.copy(), zeta, s.copy()

            search_iter = 0
            res_norm_new = 2*res_norm
            while res_norm_new > res_norm and search_iter < 50:  # Line search for the step size
                search_iter += 1

                # Update the solutions
                x = x_old + step*incr_x
                _lambda = lambda_old + step*incr_lambda
                xi = xi_old + step*incr_xi
                eta = eta_old + step*incr_eta
                if comm.rank == 0:
                    y = y_old + step*incr_y
                    z = z_old + step*incr_z
                    mu = mu_old + step*incr_mu
                    zeta = zeta_old + step*incr_zeta
                    s = s_old + step*incr_s

                # Evaluate the residual norm
                p_lambda = p0 + _lambda@P_mat
                q_lambda = q0 + _lambda@Q_mat
                g_vec = P_mat@(1.0/(upp-x)) + Q_mat@(1.0/(x-low))
                g_vec = comm.allreduce(g_vec, op=MPI.SUM)
                dpsi_dx = p_lambda/(upp-x)**2 - q_lambda/(x-low)**2

                res_x = comm.gather(dpsi_dx-xi+eta, root=0)
                res_xi = comm.gather(xi*(x-alpha)-eps, root=0)
                res_eta = comm.gather(eta*(beta-x)-eps, root=0)
                if comm.rank == 0:
                    res_x, res_xi, res_eta = np.hstack(res_x), np.hstack(res_xi), np.hstack(res_eta)
                    res_y = c + d*y - mu - _lambda
                    res_z = a0 - zeta - a@_lambda
                    res_lambda = g_vec - a*z - y + s - b
                    res_mu = mu*y - eps
                    res_zeta = zeta*z - eps
                    res_s = _lambda*s - eps
                    res = np.hstack([res_x, res_y, res_z, res_lambda, res_xi,
                                    res_eta, res_mu, res_zeta, res_s])
                    res_norm_new = np.linalg.norm(res, ord=2)
                else:
                    res_norm_new = None
                res_norm_new = comm.bcast(res_norm_new, root=0)
                step *= 0.5

            res_norm = res_norm_new
            res_max = np.linalg.norm(res, ord=np.inf) if comm.rank == 0 else None
            res_max = comm.bcast(res_max, root=0)
        eps *= 0.1

    return x
