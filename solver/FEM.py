
import numpy as np
import sympy as sym
import math
import matplotlib.pyplot as plt # Matplotlib imported for plotting, tri is for the triangulation plots
import matplotlib.tri as tri
from . import interpolator
from . import error

def FEM_solver(geometry, physics):
    """Function that returns the functions you need to calculate stiffness and mass matrix and the error.

    """
    coordinates = geometry["coordinates"]
    elements = geometry["elements"]
    boundary_elements_dirichlet = geometry["boundary_elements_dirichlet"]
    boundary_elements_neumann = geometry["boundary_elements_neumann"]

    boundary_nodes_dirichlet = set()
    for e in boundary_elements_dirichlet:
        boundary_nodes_dirichlet.add(e[0])
        boundary_nodes_dirichlet.add(e[1])





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
    def quad_2d_2nd_order_shape(ele_num, f, loc_node):
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

    local_interpolator = interpolator.LocalInterpolator(elements,boundary_elements_dirichlet,local_to_reference_map)

    A = np.zeros((len(coordinates),len(coordinates)))
    B = np.zeros((len(coordinates),len(coordinates)))
    Z = np.zeros((len(coordinates),len(coordinates)))
    #Shape function integrals:
    shape_int = []
    for fun1 in shape_func:
        tmp = []
        for fun2 in shape_func:
            tmp.append(sym.integrate(fun1*fun2,(y,0,1-x),(x,0,1)))
        shape_int.append(tmp)

    e_z_global = np.array([[0],[1]])

    def mass(f=None,u=None):
        """Returns the stiffness matrix, <phi_i, phi_j>, if nothing is passed, otherwise it returns the inner product <f(u),phi_i>
            where phi_i are the basis hat functions.
        """
        if u is None:
            B = np.zeros((len(coordinates),len(coordinates)))
            for e in range(len(elements)):
                # extract element information
                [J,c] = local_to_reference_map(e)
                jac = np.linalg.det(J) #Determinant of tranformation matrix = inverse of area of local elements
                #Local assembler
                for j in range(3):
                    for i in range(3):
                        if elements[e][i] in boundary_nodes_dirichlet or elements[e][j] in boundary_nodes_dirichlet:
                            pass
                        else:
                            B[elements[e][i]][elements[e][j]] += shape_int[i][j]*1/jac
            return B
        M = np.zeros(len(coordinates))
        for e in range(len(elements)):
            # extract element information
            [J,c] = local_to_reference_map(e)
            transform = J.dot(J.transpose()) #J*J^t; derivative transformation
            jac = np.linalg.det(J) #Determinant of tranformation matrix = inverse of area of local elements
            interpolator = lambda x,y: local_interpolator(f,u,e,x,y)
            #Local assembler
            for j in range(3):
                if elements[e][j] in boundary_nodes_dirichlet:
                    pass
                else:
                    M[elements[e][j]] = float(M[elements[e][j]]) + quad_2d_2nd_order_shape(e,interpolator,j)/jac


        return(M)
    def source(t=None,u=None,K=None):
        """Returns the inner product <source(x,y,t),phi_i>+<K(psi)e_z,phi_i>
        """
        f_vect = np.zeros((len(coordinates)))
        f_old = physics["source"]
        neumann_boundary = lambda x,y: physics["neumann"](x,y,t)
        if t is None:
            f = lambda x,y: f_old(x,y)
        else:
            f = lambda x,y: f_old(x,y,t)

        for e in range(len(elements)):
            # extract element information
            [J,c] = local_to_reference_map(e)
            jac = np.linalg.det(J) #Determinant of tranformation matrix = inverse of area of local elements
            #Local assembler
            for j in range(3):
                f_vect[elements[e][j]] = float(f_vect[elements[e][j]]) + quad_2d_2nd_order_shape(e,f,j)/jac       
            #for computing gravity
            if u is not None:
                interpolator = lambda x,y: local_interpolator(K,u,e,x,y)
                [R,d] = reference_to_local_map(e)
                x_1 = R.dot(np.array([[1/6],[1/6]]))+d
                x_2 = R.dot(np.array([[1/6],[2/3]]))+d
                x_3 = R.dot(np.array([[2/3],[1/6]]))+d
                w_1 = 1/6; w_2 = 1/6; w_3 = 1/6
                K_int = (w_1*interpolator(x_1[0][0],x_1[1][0])+w_2*interpolator(x_2[0][0],x_2[1][0])+w_3*interpolator(x_3[0][0],x_3[1][0]))
                for j in range(3):
                    f_vect[elements[e][j]]=float(f_vect[elements[e][j]]) #- K_int*e_z_global.T.dot(J.T.dot(shape_grad[j]))/jac


        for e in range(len(boundary_elements_neumann)):
            for i in range(2):
                f_vect[boundary_elements_neumann[e][i]] = f_vect[boundary_elements_neumann[e][i]] + quad_2nd_ord_line(neumann_boundary,e,i)
        for e in range(len(boundary_elements_dirichlet)):
            f_vect[boundary_elements_dirichlet[e][0]]=physics["dirichlet"](coordinates[boundary_elements_dirichlet[e][0]][0], coordinates[boundary_elements_dirichlet[e][0]][1],t)
            f_vect[boundary_elements_dirichlet[e][1]]=physics["dirichlet"](coordinates[boundary_elements_dirichlet[e][1]][0], coordinates[boundary_elements_dirichlet[e][1]][1],t)
        return f_vect
    def stiffness(f,u):
        """ Returns <f(u)grad(phi_i),grad(phi_j)>, if one just wants the simple stiffness matrix
            One would just pass f=lambda x:1
        """
        C = np.zeros((len(coordinates),len(coordinates)))
        # Matrix assembly
        for e in range(len(elements)):
            interpolator = lambda x,y: local_interpolator(f,u,e,x,y)
            # extract element information
            [J,c] = local_to_reference_map(e)
            [R,d] = reference_to_local_map(e)
            transform = J.dot(J.transpose()) #J*J^t; derivative transformation
            jac = np.linalg.det(J) #Determinant of tranformation matrix = inverse of area of local elements

            #Local assembler
            for j in range(3):
                for i in range(3):
                    x_1 = R.dot(np.array([[1/6],[1/6]]))+d
                    x_2 = R.dot(np.array([[1/6],[2/3]]))+d
                    x_3 = R.dot(np.array([[2/3],[1/6]]))+d
                    w_1 = 1/6; w_2 = 1/6; w_3 = 1/6
                    K_int = (w_1*interpolator(x_1[0][0],x_1[1][0])+w_2*interpolator(x_2[0][0],x_2[1][0])+w_3*interpolator(x_3[0][0],x_3[1][0]))
                    C[elements[e][i]][elements[e][j]] += K_int*(shape_grad[i]).transpose().dot(transform.dot(shape_grad[j]))/jac

            #Dirichlet
        for e in range(len(boundary_elements_dirichlet)):
            C[boundary_elements_dirichlet[e][0],:]=0
            C[boundary_elements_dirichlet[e][0]][boundary_elements_dirichlet[e][0]]=1
            C[boundary_elements_dirichlet[e][1],:]=0
            C[boundary_elements_dirichlet[e][1]][boundary_elements_dirichlet[e][1]]=1

        return C
    return (mass,stiffness,source,error.Error(local_interpolator,elements,coordinates))



def plot(u, geometry):
    xcoords, ycoords = geometry["coordinates"].T
    plt.tricontourf(xcoords, ycoords, geometry["elements"], u,25)
    plt.colorbar()
    plt.show()
def vectorize(u,geometry):
    """ interpolates a function so one gets it's coordinates in the solution space expressed in the basis hat functions.
    """
    x = sym.symbols('x')
    y = sym.symbols('y')
    coordinates = geometry["coordinates"]
    u_vect = np.zeros((len(coordinates)))
    for i in range(len(u_vect)):
        u_vect[i] = u.subs(x,coordinates[i][0]).subs(y,coordinates[i][1])
    u_vect = u_vect.squeeze()
    return u_vect
    