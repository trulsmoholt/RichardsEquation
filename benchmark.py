import numpy as np
import sympy as sym
import math
from FEM import FEM_solver, plot,vectorize
from richards import Richards
import matplotlib.pyplot as plt
theta_s = 0.396
theta_r = 0.131
k_s = 0.0496
alpha = 0.423
n = 2.06

s_wm = 0.125
L_s = 1.33
d = 0.05947441217


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
for psi in p_range:
    print("presure: ",psi," conductivity:",K(psi))
axs[0].set(xlabel='pressure',ylabel = 'conductivity')
axs[1].plot(p_range,list(map(theta,p_range)),'-*')
for psi in p_range:
    print("pressure: ",psi,"saturation: ",theta(psi))
axs[1].set(xlabel='pressure',ylabel = 'saturation')

plt.show()


x = sym.Symbol('x')
y = sym.Symbol('y')
t = sym.Symbol('t')
p = sym.Symbol('p')
dt = 1/16



dirichlet = sym.Piecewise(
    (1-y, y<=1),
    (-2 + 2.2*t/dt,y>1)
)



f = 0
f = sym.lambdify([x,y,t],f)

equation = Richards()
physics = equation.getPhysics()
physics['source'] = f
physics['neumann'] = lambda x,y,t: 0
physics['dirichlet'] = sym.lambdify([x,y,t],dirichlet)
mass,A,B,stiffness,source,u,error = FEM_solver(equation.geometry, equation.physics, initial = True)

th = np.linspace(0,3/16,9,endpoint=False)
print(th)
tau = th[1]-th[0]
TOL = 0.0005
L = 0.04501
u_j = np.zeros(u.shape)
u_j_n = np.ones(u.shape)

u_init = 1-y

u = vectorize(u_init,equation.geometry)
plot(u,equation.geometry)

def newton(u_j,u_n,TOL,L,K,tau,f):
    rhs = L*B@u_j+mass(theta,u_n)-mass(theta,u_j)+tau*f
    A = stiffness(K,u_j,gravity=True)
    lhs = L*B+tau*A
    u_j_n = np.linalg.solve(lhs,rhs)

    print(np.linalg.norm(u_j_n-u_j))

    if np.linalg.norm(u_j_n-u_j)>TOL + TOL*np.linalg.norm(u_j_n):
        return newton(u_j_n,u_n,TOL,L,K,tau,f)
    else:
        return u_j_n


for i in th[1:]:
    u =  newton(u,u,TOL,L,K,tau,source(i))
    plot(u,equation.geometry)
    if i>dt:
        dirichlet = sym.Piecewise(
            (1-y, y<=1),
            (0.2 , y>1)
        )
        physics['dirichlet'] = sym.lambdify([x,y,t],dirichlet)



