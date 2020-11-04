import numpy as np
import sympy as sym
from FEM import FEM_solver, plot
from richards import Richards
from parametrization import theta
x = sym.Symbol('x')
y = sym.Symbol('y')
t = sym.Symbol('t')
u_exact = t*x*y*(x-1)*(y-1)

f = (-3*sym.Pow(u_exact,2)+3*u_exact)*sym.diff(u_exact,t)
f = sym.lambdify([x,y,t],f)
print(f(0.5,0.5,1))
print(f)

equation = Richards()
mass,B,A,source,u = FEM_solver(equation.geometry, equation.physics, initial = True)

t = np.linspace(0,1,10)

tau = t[1]-t[0]
TOL = 0.00005
L = 0.15
K = 0.025
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
        return u_j_n


for i in t:
    u =  newton(u,u,TOL,L,K,tau)
    plot(u,equation.geometry)






