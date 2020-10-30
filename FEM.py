
import numpy as np
import sympy as sym
import math
import matplotlib.pyplot as plt # Matplotlib imported for plotting, tri is for the triangulation plots
import matplotlib.tri as tri

def FEM_solver(geometry, physics, initial = False):
    coordinates = geometry["coordinates"]
    elements = geometry["elements"]
    boundary_elements_dirichlet = geometry["boundary_elements_dirichlet"]
    boundary_elements_neumann = geometry["boundary_elements_neumann"]


    #Spatial coordinates
    x = sym.symbols('x')
    y = sym.symbols('y')

    # For paramterizing line integrals
    s = sym.symbols('s')
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
        return (1/6)*(f(x_1[0][0],x_1[1][0])*shape_func[loc_node].subs([(x,1/6),(y,1/6)])+f(x_2[0][0],x_2[1][0])*shape_func[loc_node].subs([(x,2/3),(y,1/6)])+f(x_3[0][0],x_3[1][0])*shape_func[loc_node].subs([(x,1/6),(y,2/3)]))
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
        return g(x_1,y_1)*dr



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

    e_z_global = np.array([[0],[1]])
    # Matrix assembly
    for e in range(len(elements)):
        # extract element information
        [J,c] = local_to_reference_map(e)
        e_z = J.dot(e_z_global)
        transform = J.dot(J.transpose()) #J*J^t; derivative transformation
        jac = np.linalg.det(J) #Determinant of tranformation matrix = inverse of area of local elements
        #Local assembler
        for j in range(3):
            u[elements[e][j]] = physics["initial"](coordinates[elements[e][j]][0],coordinates[elements[e][j]][1])
            f_vect[elements[e][j],0] = float(f_vect[elements[e][j],0]) + quad_2d_2nd_order(e,physics["source"],j)/jac
            for i in range(3):
                A[elements[e][i]][elements[e][j]] += 0.5*(shape_grad[i]+e_z).transpose().dot(transform.dot(shape_grad[j]))/jac
                B[elements[e][i]][elements[e][j]] += shape_int[i][j]*1/jac
    #Neumann
    for e in range(len(boundary_elements_neumann)):
        for i in range(2):
            f_vect[boundary_elements_neumann[e][i]] = f_vect[boundary_elements_neumann[e][i]] + quad_2nd_ord_line(physics["neumann"],e,i)

    #Dirichlet
    for e in range(len(boundary_elements_dirichlet)):
        A[boundary_elements_dirichlet[e][0],:]=0
        A[boundary_elements_dirichlet[e][0]][boundary_elements_dirichlet[e][0]]=1
        A[boundary_elements_dirichlet[e][1],:]=0
        A[boundary_elements_dirichlet[e][1]][boundary_elements_dirichlet[e][1]]=1
        f_vect[boundary_elements_dirichlet[e][0]]=physics["dirichlet"](coordinates[boundary_elements_dirichlet[e][0]][0], coordinates[boundary_elements_dirichlet[e][0]][1])
        f_vect[boundary_elements_dirichlet[e][1]]=physics["dirichlet"](coordinates[boundary_elements_dirichlet[e][1]][0], coordinates[boundary_elements_dirichlet[e][1]][1])
            
    def mass(f=None,u=None):
        def local_interpolator(f,u,ele_num,x,y):
            J,c = local_to_reference_map(ele_num)
            x_r = J@np.array([[float(x),float(y)]]).T
            return f(psi[elements[ele_num][0]]*(1-x_r[0]-x_r[1])+u[elements[ele_num][1]]*x_r[0]+u[elements[ele_num][2]]*x_r[1])
        B = np.zeros(f_vect.shape[0])
        for e in range(len(elements)):
            # extract element information
            [J,c] = local_to_reference_map(e)
            e_z = J.dot(e_z_global)
            transform = J.dot(J.transpose()) #J*J^t; derivative transformation
            jac = np.linalg.det(J) #Determinant of tranformation matrix = inverse of area of local elements
            interpolator = lambda x,y: local_interpolator(f,u,e,x,y)
            #Local assembler
            for j in range(3):
                B[elements[e][j]] = float(B[elements[e][j]]) + quad_2d_2nd_order(e,interpolator,j)/jac
        return(B)

                    

    
    if initial:
        return(mass,A,f_vect,u)
    return (mass,A,f_vect)

def plot(u, geometry):
    #Plot plot solution
    xcoords, ycoords = geometry["coordinates"].T
    plt.tricontourf(xcoords, ycoords, geometry["elements"], u)
    plt.colorbar()
    plt.show()

