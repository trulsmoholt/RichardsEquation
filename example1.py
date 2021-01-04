"""An example of pumping and suction from A study on iterative methods for solving Richards' equation by Florian List and Florin A. Radu 2016
"""

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
import math

#for solving
from solver.FEM import FEM_solver, plot,vectorize
from richards import Richards

#soil parameters
theta_s = 0.42
theta_r = 0.026
k_s = 0.12
alpha = 0.95
n = 2.9

psi_vad = -2

def theta(psi):
    if psi <=0:
        return theta_r+(theta_s-theta_r)*math.pow(1/(1+math.pow(-alpha*psi,n)),(n-1)/n)
    else:
        return theta_s

def K(psi):
    if psi <= 0:
        return k_s*math.pow(theta(psi),0.5)*(1-math.pow(1-math.pow(theta(psi),n/(n-1)),(n-1)/n))**2
    else:
        return k_s
p_range = np.linspace(-2,1,40)
fig, axs = plt.subplots(2)
axs[0].plot(p_range,list(map(K,p_range)),'-*')
axs[0].set(xlabel='pressure',ylabel = 'conductivity')
axs[1].plot(p_range,list(map(theta,p_range)),'-*')

axs[1].set(xlabel='pressure',ylabel = 'saturation')

plt.show()


x = sym.Symbol('x')
y = sym.Symbol('y')
t = sym.Symbol('t')


f = 0.006*sym.cos(4/3*math.pi*y)*sym.sin(2*math.pi*x)
f = sym.lambdify([x,y,t],f)

equation = Richards(mesh_path="mesh/example1.msh")
physics = equation.getPhysics()
physics['source'] = f
physics['neumann'] = lambda x,y,t: 0
physics['dirichlet'] = lambda x,y,t:psi_vad
mass,stiffness,source,error = FEM_solver(equation.geometry, equation.physics)
B = mass()
th = np.linspace(0,10,10,endpoint=False)
print(th)
tau = th[1]-th[0]
TOL = 0.00001
L = 0.15
u_j = np.zeros(B.shape[1])
u_j_n = np.ones(B.shape[1])

u_init = sym.Piecewise(
    (psi_vad, y>=-3/4),
    (-y-3/4,y<-3/4)
)

u = vectorize(u_init,equation.geometry)
plot(u,equation.geometry)

def L_scheme(u_j,u_n,TOL,L,K,tau,f,num_iter):
    num_iter = num_iter + 1
    rhs = L*B@u_j+mass(theta,u_n)-mass(theta,u_j)+tau*f
    A = stiffness(K,u_j,gravity=True)
    lhs = L*B+tau*A
    u_j_n = np.linalg.solve(lhs,rhs)

    print('error: ',np.linalg.norm(u_j_n-u_j),'iteration: ',num_iter)

    if np.linalg.norm(u_j_n-u_j)>TOL + TOL*np.linalg.norm(u_j_n):
        return L_scheme(u_j_n,u_n,TOL,L,K,tau,f,num_iter)
    else:
        return u_j_n

for i in th[1:]:
    num_iter = 0
    u =  L_scheme(u,u,TOL,L,K,tau,source(i),num_iter)

    plot(u,equation.geometry)


