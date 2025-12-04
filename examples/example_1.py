# Example of multiparametric QP-GNEP problem.
#
# (C) A. Bemporad, S. Hall, 2025

import numpy as np
from matplotlib import pyplot as plt
from nash_mpqp import NashMPQP

np.random.seed(0)

N = 2  # number of agents
dim = [1,1]  # number of variables for each agent
npar = 2  # number of parameters
ncon = 1  # number of constraints

#split = None
#split = "variational"
split = "min-norm"
#split = "welfare"
#split = "variational-split"

parallel_processing = True  # If True, solves the mpQPs in parallel using all available CPU cores
verbose = True
savefigs = True

# Create problem data
nvar = sum(dim)  # total number of variables
Q, c, F = [[],[],[]]

for i in range(N):
    if i==0:
        Q.append(np.array([[1.0, -1.0], [-1.0, 0.0]]))  
        F.append(np.array([[0.0, 1.0], [0.0, 0.0]]))
        
    if i==1:
        Q.append(np.array([[0.0, 1.0], [1.0, 2.0]]))
        F.append(np.zeros([nvar, npar]))
        
    c.append(np.zeros(nvar))

A = np.array([[-1.0, -1.0]])
b = np.zeros(ncon)
S = np.array([[1.0, 0.0]]) 

lb = np.zeros(nvar)  # lower bounds on variables
ub = None
xmin = lb # lower bounds on variables, used for computing the parametric solution
xmax = 10.*np.ones(nvar)   # upper bounds on variables, used for computing the parametric solution

pmin = -10*np.ones(npar) # lower bounds on parameters
pmax = 10*np.ones(npar) # upper bounds on parameters

nash_mpqp = NashMPQP(dim, pmin, pmax, xmin, xmax, Q, c, F, A, b, S, lb, ub, split=split, parallel_processing=parallel_processing, verbose=verbose)
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
