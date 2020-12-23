import numpy as np
import matplotlib.pyplot as plt
import unittest
class LocalInterpolator:
    def __init__(self,elements, boundary_nodes_dirichlet,local_to_reference_map):
        self.local_to_reference_map = local_to_reference_map
        self.elements = elements
        self.boundary_nodes_dirichlet = boundary_nodes_dirichlet

    def __call__(self,f,u,ele_num,x,y):
        J,c = self.local_to_reference_map(ele_num)
        self.determinant = np.linalg.det(J)
        x_r = J@np.array([[float(x),float(y)]]).T + c
        x2 = 1-x_r[0]-x_r[1];x1 = x_r[1];x0 = x_r[0]

        return f(u[self.elements[ele_num][2]]*x2+u[self.elements[ele_num][0]]*x0+u[self.elements[ele_num][1]]*x1)



class interpolator_test(unittest.TestCase):
    def test_simple_function(self):
        coordinates = np.array([[0,0],
                    [0,0.5],
                    [0,1],
                    [0.5,0],
                    [0.5,0.5],
                    [0.5,1],
                    [1,0],
                    [1,0.5],
                    [1,1]])

        elements = np.array([[5,2,4],
                                [2,1,4],
                                [5,4,8],
                                [1,0,4],
                                [8,4,7],
                                [4,0,3],
                                [7,4,6],
                                [4,3,6]])

        boundary_elements = np.array([[0,1],
                                        [1,2],
                                        [2,5],
                                        [5,8],
                                        [8,7],
                                        [7,6],
                                        [3,6],
                                        [0,3]])

        #map for element 0
        J = np.array([[2,2],[-2,0]])
        c = np.array([[-2],[1]])
        local_to_reference_map = lambda x: [J,c]


        u = np.array([-1,-1,-1,-1,-2,-1,-1,-1,-1])
        X,Y = coordinates.T
        plt.tricontourf(X,Y,elements,u)
        plt.colorbar()

        plt.show()
        local_interpolator = LocalInterpolator(elements,boundary_elements,local_to_reference_map)

        self.assertEqual(float(local_interpolator(lambda x:x,u,0,0.25,1)),-1)


if __name__ == "__main__":
    unittest.main()