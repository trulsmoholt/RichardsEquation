import numpy as np
import unittest
class LocalInterpolator:
    def __init__(self,elements, boundary_nodes_dirichlet,local_to_reference_map):
        self.local_to_reference_map = local_to_reference_map
        self.elements = elements
        self.boundary_nodes_dirichlet = boundary_nodes_dirichlet

    def __call__(self,f,u,ele_num,x,y):
        J,c = self.local_to_reference_map(ele_num)
        x_r = J@np.array([[float(x),float(y)]]).T + c
        x2 = 1-x_r[0]-x_r[1];x1 = x_r[1];x0 = x_r[0]
        if self.elements[ele_num][2] in self.boundary_nodes_dirichlet:
            x2 = 0
        if self.elements[ele_num][1] in self.boundary_nodes_dirichlet:
            x1=0
        if self.elements[ele_num][0] in self.boundary_nodes_dirichlet:
            x0 = 0
        return f(u[self.elements[ele_num][2]]*x2+u[self.elements[ele_num][0]]*x0+u[self.elements[ele_num][1]]*x1)


class interpolator_test(unittest.TestCase):
    #TODO write test
    def test_simple_function(self):
        self.assertEqual(True,False)


if __name__ == "__main__":
    unittest.main()