

import numpy as np
import sympy as sym
import math

from FEM import FEM_solver, plot
from richards import Richards

equation = Richards()
B,A,u = FEM_solver(equation.geometry, equation.physics, initial = True)




t = np.linspace(0,20,300)

k = t[1]-t[0]
C = 0.1*k*A+B
F = np.zeros((C.shape[0]))
for i in t:
    rhs = B@u#+f 
    u = np.linalg.solve(C,rhs)
plot(u,equation.geometry)







