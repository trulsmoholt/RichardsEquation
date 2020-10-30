

import numpy as np

from FEM import FEM_solver, plot
from richards import Richards
from parametrization import theta

equation = Richards()
mass,B,A,f,u = FEM_solver(equation.geometry, equation.physics, initial = True)

t = np.linspace(0,1,100)

tau = t[1]-t[0]
TOL = 0.25
L = 0.15
K = 0.025
counter = 0
identity = lambda x: x
v1 = mass(identity,u)
v2 = B@u
u_j = np.zeros(u.shape)
u_j_n = np.ones(u.shape)
plot(u,equation.geometry)

def newton(u_j,u_n,TOL,L,K,tau):
    rhs = L*B@u_j+mass(theta,u_n)-mass(theta,u_j)+tau*np.ndarray.flatten(f)
    lhs = L*B+tau*K*A
    u_j_n = np.linalg.solve(lhs,rhs)
    print(np.linalg.norm(u_j_n-u_j))
    if np.linalg.norm(u_j_n-u_j)>TOL + TOL*np.linalg.norm(u_j_n):
        return newton(u_j_n,u_n,TOL,L,K,tau)
    else:
        return np.linalg.solve(L*B+tau*K*A,mass(theta,u_n)-mass(theta,u_j)+tau*np.ndarray.flatten(f))


for i in t:
    u = newton(u,u,TOL,L,K,tau)
    plot(u,equation.geometry)






