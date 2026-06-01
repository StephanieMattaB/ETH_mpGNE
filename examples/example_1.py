# Example of multiparametric QP-GNEP problem.
#
# (C) A. Bemporad, S. Hall, 2025

import numpy as np
from matplotlib import pyplot as plt
from nash_mpqp import NashMPQP

np.random.seed(0)

N = 2  # number of agents
dim = [1,1]  # number of variables for each agent
# added to split parameter dependence into cost and constraints <--
npar_cost = 1  # parameters entering the cost only
npar_con  = 1  # parameters entering the constraints only
npar = npar_cost + npar_con  # p = [p_cost; p_con] 
ncon = 1  # number of constraints b_sh

#split = None
#split = "variational"
split = "min-norm"
#split = "welfare"
#split = "variational-split"

parallel_processing = True  # If True, solves the mpQPs in parallel using all available CPU cores
verbose = True
savefigs = False

# Create problem data
nvar = sum(dim)  # total number of variables
Q, c, F = [[],[],[]]

for i in range(N):
    if i==0:
        Q.append(np.array([[1.0, -1.0], [-1.0, 0.0]]))
        # F = [F_cost | F_con]: agent 0 cost depends on p_cost, not p_con
        F_cost_0 = np.array([[1.0], [0.0]])          # (nvar x npar_cost)
        F_con_0  = np.zeros((nvar, npar_con))         # (nvar x npar_con) — zeroed
        F.append(np.hstack([F_cost_0, F_con_0]))

    if i==1:
        Q.append(np.array([[0.0, 1.0], [1.0, 2.0]]))
        F.append(np.zeros([nvar, npar]))              # agent 1 cost has no parameter dependence

    c.append(np.zeros(nvar))

A = np.array([[-1.0, -1.0]])
b = np.zeros(ncon)
# S = [S_cost | S_con]: constraint depends on p_con, not p_cost
S_cost = np.zeros((ncon, npar_cost))                 # (ncon x npar_cost) — zeroed
S_con  = np.array([[1.0]])                           # (ncon x npar_con)
S = np.hstack([S_cost, S_con])

lb = np.zeros(nvar)  # lower bounds on variables
ub = None
xmin = lb # lower bounds on variables, used for computing the parametric solution
xmax = 10.*np.ones(nvar)   # upper bounds on variables, used for computing the parametric solution

pmin_cost = -10*np.ones(npar_cost)
pmax_cost =  10*np.ones(npar_cost)
pmin_con  = -10*np.ones(npar_con)
pmax_con  =  10*np.ones(npar_con)
pmin = np.hstack([pmin_cost, pmin_con])  # lower bounds on p = [p_cost; p_con]
pmax = np.hstack([pmax_cost, pmax_con])  # upper bounds on p = [p_cost; p_con]

nash_mpqp = NashMPQP(dim, pmin, pmax, xmin, xmax, Q, c, F, A, b, S, lb, ub, split=split, parallel_processing=parallel_processing, verbose=verbose)
# (n,theta[lb,up], x[lb,up] ?, Q, q, F, s.t. A, b, Btheta)
nash_mpqp.solve()

print(nash_mpqp.statistics())

plt.figure()
np.random.seed(1)
nash_mpqp.plot_2d(show_centers=True, show_circles=True, show_legend=True)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.title("")
plt.legend(loc='center right', fontsize=12)
plt.show()

# Verify equilibrium conditions
nash_mpqp.check_nash_equilibria()
