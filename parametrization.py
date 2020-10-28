import numpy as np
import matplotlib.pyplot as plt
import math
theta_s = 0.42
theta_r = 0.026
k_s = 0.12
alpha = 0.95
n = 2.9



def theta(psi):
    t = np.zeros(psi.shape)
    for i in range(psi.shape[0]):
        print(psi[i])
        if psi[i] <=0:
            t[i]=theta_r+(theta_s-theta_r)*math.pow(1/(1+math.pow(-alpha*psi[i],n)),(n-1)/n)
        else:
            t[i]=theta_s
    return t

def k(psi):
    return k_s*math.pow(theta(psi),0.5)*(1-math.pow(1-math.pow(theta(psi),n/(n-1)),(n-1)/n)**2)




# x = np.linspace(-3,0,20,endpoint=False)
# for t in x:
#     plt.plot(t,k(t),'*')
# plt.show()