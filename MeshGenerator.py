import numpy as np
import math
import unittest
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
class Mesh:
    def __init__(self, max_h: float):
        self.max_h = max_h
        self.X = 1
        self.Y = 1
        self.num_boundary_elements = math.ceil(self.X/(max_h/math.sqrt(2))) #number of elements on one side of domain
        self.coordinates = np.array([[0,0]])
        self.__generate_coordinates()
        #self.__generate_elements()
        self.__generate_boundary_elements()
        self.elements = Delaunay(self.coordinates).simplices
    def __generate_coordinates(self):
        xp = np.linspace(0,self.X,self.num_boundary_elements+1)
        yp = np.linspace(0,self.Y,self.num_boundary_elements+1)
        coordinates = np.array([[0,0]])
        elements = np.array([[0,0,0]])
        for i in range(self.num_boundary_elements+1):
            for j in range(self.num_boundary_elements+1):
                coordinates = np.concatenate((coordinates,np.array([[xp[i],yp[j]]])))
        self.coordinates=np.delete(coordinates,0,0)
    def __generate_elements(self):
        """Makes a vector of elements, where each element is an index in the coordinate vector.
        """
        def get_element_top_left(n: int):
            """Return element as row vector of three indices pointing to coordinate vector.

            Keyword arguments:
            n -- index of top left corner of triangle
            """
            return np.array([n,n-self.num_boundary_elements-1,n+1])
        def get_element_bottom_right(n: int):
            return np.array([n,n+self.num_boundary_elements+1,n-1])


        elements = np.array([[0,0,0]])
        top_left = filter(lambda x: (x%self.num_boundary_elements+2) != 0,range(self.num_boundary_elements+2,len(self.coordinates)))
        bottom_right = filter(lambda x: x%self.num_boundary_elements != 1,range(self.num_boundary_elements+2,len(self.coordinates)))
        for i,j in zip(top_left,bottom_right):
            elements = np.concatenate((elements,np.array([get_element_top_left(i)])))
            elements = np.concatenate((elements,np.array([get_element_top_left(j)])))
        self.elements = np.delete(elements,0,0)


    def __generate_boundary_elements(self):
        boundary_elements = np.array([[0,0]])
        for i in range(0,self.num_boundary_elements):
            boundary_elements = np.concatenate((boundary_elements,np.array([[i,i+1]])))#bottom
            top_bnd = len(self.coordinates)-self.num_boundary_elements-1
            boundary_elements = np.concatenate((boundary_elements,np.array([[top_bnd + i,top_bnd +i+1]])))#top
            boundary_elements = np.concatenate((boundary_elements,np.array([[i*(self.num_boundary_elements+1),(i+1)*(self.num_boundary_elements+1)]])))#left
            boundary_elements = np.concatenate((boundary_elements,np.array([[(i+1)*(self.num_boundary_elements+1)-1,(i+2)*(self.num_boundary_elements+1)-1]])))#right
            self.boundary_elements = np.delete(boundary_elements,0,0)           
# mesh = Mesh(0.5*math.sqrt(2))
# points = mesh.coordinates
# print(mesh.elements)
# print(mesh.coordinates)

# plt.triplot(points[:,0], points[:,1], mesh.elements)
# plt.plot(points[:,0], points[:,1], 'o')
# plt.show()


class TestMesh(unittest.TestCase):
    def test_boundary_elements(self):
        res = np.array([[0, 1],
                        [6, 7],
                        [0, 3],
                        [2, 5],
                        [1, 2],
                        [7, 8],
                        [3, 6],
                        [5, 8]])


        mesh = Mesh(0.5*math.sqrt(2))
        self.assertEqual(mesh.boundary_elements.all(),res.all())
    def test_2(self):
        mesh = Mesh(0.34*math.sqrt(2))
        print(mesh.boundary_elements)
        print(len(mesh.coordinates))
        print(mesh.num_boundary_elements)
        points = mesh.coordinates
        plt.triplot(points[:,0], points[:,1], mesh.elements)
        plt.plot(points[:,0], points[:,1], 'o')
        plt.show()
if __name__ == "__main__":
    unittest.main()