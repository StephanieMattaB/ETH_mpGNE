# Example of multiparametric QP-GNEP problem.
#
# (C) A. Bemporad, S. Hall, 2025

import numpy as np
import matplotlib.pyplot as plt
from nash_mpqp import NashMPQP

np.random.seed(2)

dim = [2,3,2]  # number of variables for each agent
N = len(dim) # number of agents
npar = 2  # number of parameters
ncon = 2  # number of constraints
#split = "variational"
#split = "fair"
#split = "min-norm"
split = None
parallel_processing = True  # If True, solves the mpQPs in parallel using all available CPU cores
verbose = False  # If True, print additional progress information

# Create problem data
nvar = sum(dim)  # total number of variables
Q, c, F = [[],[],[]]
for i in range(N):
    # each agent has its own quadratic cost function
    Qi = np.random.randn(nvar,nvar); Qi=Qi@Qi.T + 1.e-3*np.eye(nvar)
    Q.append(Qi)
    c.append(np.random.randn(nvar))
    F.append(np.random.randn(nvar,npar))
A = np.random.randn(ncon,nvar)
b = np.random.rand(ncon)
S = np.random.randn(ncon,npar)  
lb = None # no lower bounds on variables
ub = None # no upper bounds on variables
xmin = -100.*np.ones(nvar)  # lower bounds on variables, just used for computing the parametric solution
xmax = 100.*np.ones(nvar)   # upper bounds on variables, just used for computing the parametric solution

pmin=-10.*np.ones(npar) # lower bounds on parameters
pmax=10.*np.ones(npar) # upper bounds on parameters

nash_mpqp = NashMPQP(dim, pmin, pmax, xmin, xmax, Q, c, F, A, b, S, lb, ub, split=split, parallel_processing=parallel_processing, verbose=verbose)

# Compute the parametric solution
nash_mpqp.solve()

# Print statistics about the solution
nash_mpqp.statistics()

# Plot the parametric solution in 2D parameter space
nash_mpqp.plot_2d(show_circles=True, show_legend = True, colors = plt.get_cmap('tab10').colors)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend(loc='lower right')

# Check the computed Nash equilibria at the Chebyshev center of each critical region
nash_mpqp.check_nash_equilibria()

