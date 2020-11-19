import numpy as np
import math
class Richards:
    def __init__(self):
        #Read input file
        gmshfile = open("mesh/mesh1.msh",'r')
        nodes = False
        element = False
        physicalname = False
        coordinates = np.array([[0,0]])
        elements = np.array([[0,0,0]])
        boundary_elements_dirichlet = np.array([[0,0]])
        boundary_elements_neumann = np.array([[0,0]])
        neumannmarker = 0
        dirichletmarker = 0
        for line in gmshfile.readlines():
            if line == '$Nodes\n':
                nodes = True
            if line == '$EndNodes\n':
                nodes = False
            if line == '$Elements\n':
                element = True
            if line == '$EndElements\n':
                element = False
            if line == '$PhysicalNames\n':
                physicalname = True
            if line == '$EndPhysicalNames\n':
                physicalname = False
            
            if physicalname:
                line_elements = line.split()
                if len(line_elements)>=3:
                    if line_elements[2]=='"NeumannBdry"':
                        neumannmarker = int(line_elements[1])
                    if line_elements[2]=='"DirichletBdry"':
                        dirichletmarker = int(line_elements[1])
            

            #Creating list of coordinates
            if nodes:
                line_elements=line.split()
                if len(line_elements)>=2:
                    coordinates = np.concatenate((coordinates,np.array([[float(line_elements[1]),float(line_elements[2])]])))
            
            #Creating list of elements
            if element:
                line_elements = line.split()
                if (len(line_elements) >= 2) and (line_elements[1]=='2'):
                    elements = np.concatenate((elements,np.array([[int(line_elements[5])-1,int(line_elements[6])-1,int(line_elements[7])-1]])))
                if (len(line_elements) >= 2) and (line_elements[1]=='1') and (line_elements[3] == str(dirichletmarker)):
                    boundary_elements_dirichlet = np.concatenate((boundary_elements_dirichlet,np.array([[int(line_elements[5])-1,int(line_elements[6])-1]])))
                if (len(line_elements) >= 2) and (line_elements[1]=='1') and (line_elements[3] == str(neumannmarker)):
                    boundary_elements_neumann = np.concatenate((boundary_elements_neumann,np.array([[int(line_elements[5])-1,int(line_elements[6])-1]])))

        
        coordinates=np.delete(coordinates,0,axis=0)
        elements=np.delete(elements,0,axis=0)   
        boundary_elements_dirichlet=np.delete(boundary_elements_dirichlet,0,axis=0)
        boundary_elements_neumann=np.delete(boundary_elements_neumann,0,axis=0)
        def initial(x,y):
            return 0

        def source(x,y,t):
            return -2*t*x*(x - 1) - 2*t*y*(y - 1) + x*y*(1 - x)*(1 - y)

        def neumann(x,y,t):
            # need to multiply flux with K
            return 10 *t*x*(x - 1)
        def dirichlet(x,z):
            return -1

        self.geometry = {
            "coordinates": coordinates,
            "elements": elements,
            "boundary_elements_dirichlet": boundary_elements_dirichlet,
            "boundary_elements_neumann": boundary_elements_neumann
        }
        self.physics = {
            "initial": initial,
            "source": source,
            "neumann": neumann,
            "dirichlet": dirichlet
        }
    def getGeometry(self):
        return self.geometry
    def getPhysics(self):
        return self.physics


