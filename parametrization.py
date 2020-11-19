import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sym
theta_s = 0.42
theta_r = 0.026
k_s = 0.12
alpha = 0.95
n = 2.9

s_wm = 0.125
L_s = 1.33
d = 0.05947441217









# def theta(psi):
#     if psi <=0:
#         return theta_r+(theta_s-theta_r)*math.pow(1/(1+math.pow(-alpha*psi,n)),(n-1)/n)
#     else:
#         return theta_s

# def theta_sym():
#     p = sym.Symbol('p')
#     theta_s = sym.Piecewise(
#         ((L_s-s_wm)*d,p<0),
#         (1,p>1),
#         (s_wm*p + (L_s -s_wm)*(-(4/3)*p**3 + 2*p**2 + d),True)
#     )
#     return theta_s
def theta_sym():
    p = sym.Symbol('p')

    return 1/(1-p)



# def theta(p):
#     if p <=0:
#         return (L_s-s_wm)*d
#     elif 0<p<1:
#         return s_wm*p + (L_s -s_wm)*(-(4/3)*p**3 + 2*p**2 + d)
#     else:
#         return 1
def theta(p):
    return 1/(1-p)

def k(psi):
    return k_s*math.pow(theta(psi),0.5)*(1-math.pow(1-math.pow(theta(psi),n/(n-1)),(n-1)/n)**2)


# p = np.arange(-1,2,0.1)
# for i in p:
#     plt.plot(i,theta(i),'*')
# plt.show()