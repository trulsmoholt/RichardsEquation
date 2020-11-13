import numpy as np
import sympy as sym
from FEM import FEM_solver, plot,vectorize
from richards import Richards
from parametrization import theta,theta_sym
x = sym.Symbol('x')
y = sym.Symbol('y')
t = sym.Symbol('t')
p = sym.Symbol('p')
u_exact = t*x*y*(x-1)*(y-1)
K = 10

neumann = sym.diff(u_exact,y)
print(neumann.subs(y,1))
theta_s = theta_sym()
f = (K*(-sym.diff(u_exact,x,2)-sym.diff(u_exact,y,2)) + sym.diff(theta_s.subs(p,u_exact),t))

f = sym.lambdify([x,y,t],f)


equation = Richards()
physics = equation.getPhysics()
physics['source'] = f
mass,B,A,source,u = FEM_solver(equation.geometry, equation.physics, initial = True)

th = np.linspace(7.9,8,10)

tau = th[1]-th[0]
TOL = 0.00005
L = 0.7
u_j = np.zeros(u.shape)
u_j_n = np.ones(u.shape)

u = vectorize(u_exact.subs(t,7.9),equation.geometry)

def newton(u_j,u_n,TOL,L,K,tau,f):
    rhs = L*B@u_j+mass(theta,u_n)-mass(theta,u_j)+tau*f
    lhs = L*B+tau*K*A
    u_j_n = np.linalg.solve(lhs,rhs)
    print(np.linalg.norm(u_j_n-u_j))

    if np.linalg.norm(u_j_n-u_j)>TOL + TOL*np.linalg.norm(u_j_n):
        return newton(u_j_n,u_n,TOL,L,K,tau,f)
    else:
        return u_j_n


for i in th[1:]:
    plot(source(i),equation.geometry)
    u =  newton(u,u,TOL,L,K,tau,source(i))
    plot(u,equation.geometry)
    u_e = vectorize(u_exact.subs(t,i),equation.geometry)
    plot(u_e,equation.geometry)
    plot(u-u_e,equation.geometry)





