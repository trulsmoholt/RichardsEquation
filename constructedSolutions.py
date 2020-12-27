import numpy as np
import sympy as sym
from FEM import FEM_solver, plot,vectorize
from richards import Richards
from parametrization import theta_sym
from differentiation import gradient, divergence
x = sym.Symbol('x')
y = sym.Symbol('y')
t = sym.Symbol('t')
p = sym.Symbol('p')
u_exact = -(t)*x*y*(1-x)*(1-y) - 1
def difficult():
    K = p**2
    theta = 1/(1-p)
    f = sym.diff(theta.subs(p,u_exact),t)-divergence(gradient(u_exact,[x,y]),[x,y],K.subs(p,u_exact))
    print(sym.simplify(f))
    f = sym.lambdify([x,y,t],f)

    equation = Richards()
    physics = equation.getPhysics()
    physics['source'] = f
    physics['neumann'] = sym.lambdify([x,y,t],K.subs(p,u_exact)*sym.diff(u_exact,y))
    mass,A,B,stiffness,source,u,error = FEM_solver(equation.geometry, equation.physics, initial = True)

    th = np.linspace(0,1,10)

    tau = th[1]-th[0]
    TOL = 0.00005
    L = 3

    u = vectorize(u_exact.subs(t,0),equation.geometry)
    theta = lambda x: 1/(1-x)

    K = lambda x: x**2
    def newton(u_j,u_n,TOL,L,K,tau,f):
        rhs = L*B@u_j+mass(theta,u_n)-mass(theta,u_j)+tau*f
        A = stiffness(K,u_j)
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
        u_e = vectorize(u_exact.subs(t,i),equation.geometry)
        plot(u_e,equation.geometry)
        plot(u-u_e,equation.geometry)
        error.l2_error(u,u_exact.subs(t,i))

difficult()
def heat_equation():
    T_end = 2
    f = sym.diff(u_exact,t,1)-(sym.diff(u_exact,x,2)+sym.diff(u_exact,y,2))
    f = sym.lambdify([x,y,t],f)
    equation = Richards()
    physics = equation.getPhysics()
    physics['source'] = f
    mass,A,B,stiffness,source,u,error = FEM_solver(equation.geometry, equation.physics, initial = True)
    u = vectorize(u_exact.subs(t,0),equation.geometry)
    th = np.linspace(0,T_end,8)
    tau = th[1]-th[0]
    for i in th[1:]:
        rhs = tau*source(i)+B@u
        lhs = tau*A+B
        u = np.linalg.solve(lhs,rhs)
        u_e = vectorize(u_exact.subs(t,i),equation.geometry)
        error.l2_error(u,u_exact.subs(t,i))

    plot(u,equation.geometry)
    plot(u_e,equation.geometry)

    error.l2_error(u,u_exact.subs(t,T_end))
#heat_equation()


def elliptic():
    u_exact = -x*y*(1-x)*(1-y)-1
    f = -(sym.diff(u_exact,x,2)+sym.diff(u_exact,y,2))
    f = sym.lambdify([x,y],f)
    equation = Richards()
    physics = equation.getPhysics()
    physics['source'] = f
    mass,A,B,stiffness,source,u,error = FEM_solver(equation.geometry, equation.physics, initial = True)
    u = np.linalg.solve(A,source())
    error.l2_error(u,u_exact)
    error.max_error(u,u_exact)
#elliptic()