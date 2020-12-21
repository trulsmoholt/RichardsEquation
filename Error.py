import numpy as np
import unittest
import sympy as sym
import math
class Error:
    def __init__(self,local_interpolator,elements,coordinates):
        self.local_interpolator = local_interpolator
        self.elements = elements
        self.coordinates = coordinates
    def _mid_points(self,ele_num):
        x_mid = (self.coordinates[self.elements[ele_num][0]][0] + self.coordinates[self.elements[ele_num][1]][0] + self.coordinates[self.elements[ele_num][2]][0])/3
        y_mid = (self.coordinates[self.elements[ele_num][0]][1] + self.coordinates[self.elements[ele_num][1]][1] + self.coordinates[self.elements[ele_num][2]][1])/3
        return (x_mid,y_mid)
    def l2_error(self,u,u_exact):
        x = sym.symbols('x')
        y = sym.symbols('y')
        err = 0
        for e in range(len(self.elements)):
            (x_mid,y_mid) = self._mid_points(e)
            u_exact_mid = u_exact.subs(x,x_mid).subs(y,y_mid)
            u_mid = self.local_interpolator(lambda x:x,u,e,x_mid,y_mid)
            err += (1/self.local_interpolator.determinant)*(u_exact_mid-u_mid)**2
        err = math.sqrt(err)
        print("L2 error:",err)