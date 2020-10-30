

import numpy as np

from FEM import FEM_solver, plot
from richards import Richards
from parametrization import theta

equation = Richards()
mass,A,f,u = FEM_solver(equation.geometry, equation.physics, initial = True)

t = np.linspace(0,1,100)

tau = t[1]-t[0]
TOL = 0.2
L = 1
K = 0.05
counter = 0
u_j = np.zeros(u.shape)
u_j_n = np.ones(u.shape)
plot(mass(theta,u),equation.geometry)

# def newton(u_j,u_n,TOL,L,K,tau):
#     rhs = L*B@u_j-B@u_n+tau*np.ndarray.flatten(f)+B@u_n
#     lhs = L*B + tau*K*np.diag(u_j)@A
#     u_j_n = np.linalg.solve(lhs,rhs)
#     print(np.linalg.norm(u_j_n-u_j))
#     if np.linalg.norm(u_j_n-u_j)>TOL:
#         return newton(u_j_n,u_n,TOL,L,K,tau)
#     else:
#         return u_j_n


# for i in t:
#     u = newton(u,u,TOL,L,K,tau)







