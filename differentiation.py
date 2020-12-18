import numpy as np
import sympy as sym
import unittest

def gradient(f, variables: list):
    vec = []
    for variable in variables:
        vec.append(sym.diff(f,variable))
    return vec
def divergence(gradient: list,variables: list, permability = False):
    if not permability:
        return sum(map(lambda t: sym.diff(t[0],t[1]),zip(gradient,variables)))
    else:
        return sum(map(lambda t: sym.diff(permability*t[0],t[1]),zip(gradient,variables)))
class TestDiffMethods(unittest.TestCase):
    def test_gradient(self):
        x =  sym.Symbol('x')
        y = sym.Symbol('y')
        res = [2*x,2*y]
        var = list([x,y])
        f =  x**2 + y**2
        self.assertEqual(gradient(f,var),res)
    def test_divergence(self):
        x =  sym.Symbol('x')
        y = sym.Symbol('y')
        var = [x,y]
        res = 2*x+2*y
        self.assertEqual(divergence([x**2,y**2],var),res)
    def test_divergence_permability(self):
        x =  sym.Symbol('x')
        y = sym.Symbol('y')
        var = [x,y]
        res = 3*x**2 + 2*x*y
        self.assertEqual(divergence([x**2,y**2],var,permability = x),res)
    def test_combination(self):
        z =  sym.Symbol('z')
        y = sym.Symbol('y')
        res = 6*z
        f = z**2+y**2
        self.assertEqual(divergence(gradient(f,[z,y]),[z,y],z),res)
if __name__ =='__main__':
    unittest.main()

