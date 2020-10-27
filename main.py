

import numpy as np
import sympy as sym
import math
import matplotlib.pyplot as plt # Matplotlib imported for plotting, tri is for the triangulation plots
import matplotlib.tri as tri

def initial(x,z):
    if z<-3/4:
        return -z-3/4
    else:
        return 0

def source(x,z):
    if z<-3/4:
        return 0
    else:
        return 0.006*math.cos(4/3*math.pi*z)*math.sin(2*math.pi*x)

def neumann(x,z):
    return 0
def dirichlet(x,z):
    return 0

#Spatial coordinates
x = sym.symbols('x')
y = sym.symbols('y')

# For paramterizing line integrals
s = sym.symbols('s')


u_fabric = (1-x)*(1-y)*x*y
#u_fabric = sym.sin(x)*sym.cos(y)

# Let the top boundary be the Neumann boundary, n = [0,1] is the outwards pointing unit normal vector. 
#n = np.array([[0],[1]])
#g = 2*nabla_u_fabric.transpose().dot(n)[0][0]
f = (-u_fabric.diff(x,2)-u_fabric.diff(y,2))
print(f)



#Read input file
gmshfile = open("mesh/neumann.msh",'r')
nodes = False
element = False
physicalname = False
coordinates = np.array([[0,0]])
elements = np.array([[0,0,0]])
boundary_elements_dirichlet = np.array([[0,0]])
boundary_elements_neumann = np.array([[0,0]])
neumannmarker = 0
dirichletmarker = 0



#print(gmshfile.readlines()[3])
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
print(boundary_elements_neumann)
print(boundary_elements_dirichlet)


#Shape functions
shape_func = np.array([x,y,1-y-x])
shape_grad = np.array([np.matrix([[1],[0]]),np.matrix([[0],[1]]),np.matrix([[-1],[-1]])])
# Making [-1,1] reference element for ease of quadrature appliance
shape_func_1d = np.array([0.5-0.5*x,0.5+0.5*x])





# Map to reference element, det(J) = jacobian of transformation (inverse of area of element)
def local_to_reference_map(ele_num):
    mat_coords = np.array([[coordinates[elements[ele_num][0]][0],coordinates[elements[ele_num][0]][1],1],[coordinates[elements[ele_num][1]][0],coordinates[elements[ele_num][1]][1],1],[coordinates[elements[ele_num][2]][0],coordinates[elements[ele_num][2]][1],1]])
    b1 = np.array([[1],[0],[0]])
    b2 = np.array([[0],[1],[0]])
    a1 = np.linalg.solve(mat_coords,b1)
    a2 = np.linalg.solve(mat_coords,b2)
    J = np.matrix([[a1[0][0],a1[1][0]],[a2[0][0],a2[1][0]]])
    c = np.matrix([[a1[2][0]],[a2[2][0]]])
    return [J,c]

# Map from reference element to local element
def reference_to_local_map(ele_num):
    mat_coords = np.array([[1,0,1],[0,1,1],[0,0,1]])
    b1 = np.array([[coordinates[elements[ele_num][0]][0]],[coordinates[elements[ele_num][1]][0]],[coordinates[elements[ele_num][2]][0]]])
    b2 = np.array([[coordinates[elements[ele_num][0]][1]],[coordinates[elements[ele_num][1]][1]],[coordinates[elements[ele_num][2]][1]]])
    a1 = np.linalg.solve(mat_coords,b1)
    a2 = np.linalg.solve(mat_coords,b2)
    J = np.matrix([[a1[0][0],a1[1][0]],[a2[0][0],a2[1][0]]])
    c = np.matrix([[a1[2][0]],[a2[2][0]]])
    return [J,c]

# Compute midpoint of function at local element (used in midpoint rule)
def mid_point(ele_num, f):
    x_mid = (coordinates[elements[ele_num][0]][0] + coordinates[elements[ele_num][1]][0] + coordinates[elements[ele_num][2]][0])/3
    y_mid = (coordinates[elements[ele_num][0]][1] + coordinates[elements[ele_num][1]][1] + coordinates[elements[ele_num][2]][1])/3
    return f.subs(x,x_mid).subs(y,y_mid)

#Second order quadrature: weights = 1/6, nodes = (1/6,1/6), (1/6,2/3), (2/3,1/6)
def quad_2d_2nd_order(ele_num, f, loc_node):
    [J,c] = reference_to_local_map(ele_num)
    x_1 = J.dot(np.array([[1/6],[1/6]]))+c
    x_2 = J.dot(np.array([[2/3],[1/6]]))+c
    x_3 = J.dot(np.array([[1/6],[2/3]]))+c
    return (1/6)*(source(x_1[1][0],x_1[1][0])*shape_func[loc_node].subs([(x,1/6),(y,1/6)])+source(x_2[1][0],x_2[1][0])*shape_func[loc_node].subs([(x,2/3),(y,1/6)])+source(x_3[1][0],x_3[1][0])*shape_func[loc_node].subs([(x,1/6),(y,2/3)]))

#Parametrizes straight boundary segment to [-1,1]
def param_1d_ele(ele_num):
    return np.array([[coordinates[boundary_elements_neumann[ele_num][0]][0]+(coordinates[boundary_elements_neumann[ele_num][1]][0]-coordinates[boundary_elements_neumann[ele_num][0]][0])*0.5*(s+1)],[coordinates[boundary_elements_neumann[ele_num][0]][1]+(coordinates[boundary_elements_neumann[ele_num][1]][1]-coordinates[boundary_elements_neumann[ele_num][0]][1])*0.5*(s+1)]])
    
#Calculates length of 1d interval
def param_1d_ele_derivative(ele_num):
    #return math.sqrt((coordinates[boundary_elements_neumann[ele_num][0]][0]-coordinates[boundary_elements_neumann[ele_num][0]][1])^2+(coordinates[boundary_elements_neumann[ele_num][1]][0]-coordinates[boundary_elements_neumann[ele_num][1]][1])^2)
    return 0.5*math.sqrt((coordinates[boundary_elements_neumann[ele_num][0]][0]-coordinates[boundary_elements_neumann[ele_num][1]][0])**2+(coordinates[boundary_elements_neumann[ele_num][0]][1]-coordinates[boundary_elements_neumann[ele_num][1]][1])**2)

# Second order quadrature on boundary line integral
def quad_2nd_ord_line(f,ele_num,loc_node):
    r = param_1d_ele(ele_num)
    dr = param_1d_ele_derivative(ele_num)
    x_1 = r[0][0].subs(s,-1/math.sqrt(3))
    x_2 = r[0][0].subs(s,1/math.sqrt(3))
    y_1 = r[1][0].subs(s,-1/math.sqrt(3))
    y_2 = r[1][0].subs(s,1/math.sqrt(3))
    return (f(x_1, y_1)*shape_func_1d[loc_node].subs(x,-1/math.sqrt(3))+f(x_2,y_2)*shape_func_1d[loc_node].subs(x,1/math.sqrt(3)))*dr

# First order quadrature on line segment (midpoint rule)
def quad_midpoint_1d(g,ele_num):
    r = param_1d_ele(ele_num)
    dr = param_1d_ele_derivative(ele_num)
    x_1 = r[0][0].subs(s,0)
    y_1 = r[1][0].subs(s,0)
    return g.subs(x,x_1).subs(y,y_1)*dr


A = np.zeros((len(coordinates),len(coordinates)))
B = np.zeros((len(coordinates),len(coordinates)))
Z = np.zeros((len(coordinates),len(coordinates)))
f_vect = np.zeros((len(coordinates),1))
u = np.zeros((A.shape[0]))
#Shape function integrals:
shape_int = []
for fun1 in shape_func:
    tmp = []
    for fun2 in shape_func:
        tmp.append(sym.integrate(fun1*fun2,(y,0,1-x),(x,0,1)))
    shape_int.append(tmp)

print(shape_int)

# Matrix assembly
for e in range(len(elements)):
    # extract element information
    [J,c] = local_to_reference_map(e)
    transform = J.dot(J.transpose()) #J*J^t; derivative transformation
    jac = np.linalg.det(J) #Determinant of tranformation matrix = inverse of area of local elements
    #Local assembler
    for j in range(3):
        u[elements[e][j]] = initial(coordinates[elements[e][j]][0],coordinates[elements[e][j]][1])
        f_vect[elements[e][j],0] = float(f_vect[elements[e][j],0]) + quad_2d_2nd_order(e,f,j)/jac
        for i in range(3):
            A[elements[e][i]][elements[e][j]] += 0.5*shape_grad[i].transpose().dot(transform.dot(shape_grad[j]))/jac
            B[elements[e][i]][elements[e][j]] += shape_int[i][j]*1/jac
            
# Modify matrix and vector according to boundary conditions

#Neumann
for e in range(len(boundary_elements_neumann)):
    for i in range(2):
        f_vect[boundary_elements_neumann[e][i]] = f_vect[boundary_elements_neumann[e][i]] + quad_2nd_ord_line(neumann,e,i)

#Dirichlet
for e in range(len(boundary_elements_dirichlet)):
    A[boundary_elements_dirichlet[e][0],:]=0
    A[boundary_elements_dirichlet[e][0]][boundary_elements_dirichlet[e][0]]=1
    A[boundary_elements_dirichlet[e][1],:]=0
    A[boundary_elements_dirichlet[e][1]][boundary_elements_dirichlet[e][1]]=1
    f_vect[boundary_elements_dirichlet[e][0]]=dirichlet(coordinates[boundary_elements_dirichlet[e][0]][0], coordinates[boundary_elements_dirichlet[e][0]][1])
    f_vect[boundary_elements_dirichlet[e][1]]=dirichlet(coordinates[boundary_elements_dirichlet[e][1]][0], coordinates[boundary_elements_dirichlet[e][1]][1])

t = np.linspace(0,0.6,10)

k = t[1]-t[0]
C = 0.1*k*A+B






F = np.zeros((C.shape[0]))
print(f_vect.shape)
for i in t:
    rhs = B@u  + np.ndarray.flatten(f_vect)
    u = np.linalg.solve(C,rhs)




#Compute values of exact solution on the nodes and calculate error in max-value
u_exact = np.zeros([len(u),1])
for i in range(len(u_exact)):
    u_exact[i]=u_fabric.subs(x,coordinates[i][0]).subs(y,coordinates[i][1])
u_exact=u_exact.squeeze()
error = np.amax(np.absolute(u-u_exact))




#Plot plot solution
xcoords, ycoords = coordinates.T
plt.tricontourf(xcoords, ycoords, elements, u)
plt.colorbar()
plt.show()


