"""
Created on Mon Sep 28 14:23:01 2020

@author: Grayson Schaer
"""
from scipy.signal import find_peaks 
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Vortex panel method solver and environment stepper for a 1m chord length airfoil
# @param max_num_steps - the max number of times an airfoil can be changed by an agent
# @param n_panels_per_surface - the number of panels per surface on the airfoil
# @param v_inf_test_points - n freestream velocity magnitudes in form np.array([v0, v1, v2, v3, v4, ..., vn])
# @param alpha_test_points - n angles of attack in form np.array([a0, a1, a2, a3, a4, ..., an])
# @param cl_test_points - n lift coefficients in form np.array([cl0, ..., cln])
# @param cdp_test_points - n pressure drag coefficients in form np.array([cdp0, ..., cdpn])
# @param cm4c_test_points - n moment coefficients about quater chord in form np.array([cm4c0, ..., cm4cn])
class Vortex_Panel_Solver():
    def __init__(self, max_num_steps, n_panels_per_surface, v_inf_test_points, alpha_test_points, cl_test_points, cdp_test_points, cm4c_test_points):
        
        self.max_num_steps = max_num_steps
        self.curr_step = 0
        self.num_actions = 4 * n_panels_per_surface - 2
        self.state_dimension = 2 * n_panels_per_surface + 1
        self.n_panels_per_surface = n_panels_per_surface
        self.z_dirn = np.array([np.zeros(self.n_panels_per_surface), np.zeros(self.n_panels_per_surface), np.ones(self.n_panels_per_surface)])
        self.precision = 10 # ***** MUST BE EVEN ***** #
        
        self.v_inf_test_points = v_inf_test_points
        self.alpha_test_points = alpha_test_points
        self.cl_test_points = cl_test_points
        self.cdp_test_points = cdp_test_points
        self.cm4c_test_points = cm4c_test_points
        
        # Create upper surface that is legal
        self.upper_surface_x = np.linspace(1,0,self.n_panels_per_surface+1).reshape(1, self.n_panels_per_surface+1)
        upper_surface_y = np.zeros((1,self.n_panels_per_surface+1))
        upper_surface_y[0][0] = 0.01
        highest_vertex = np.random.randint(1,self.n_panels_per_surface)
        highest_vertex_height = np.random.randint(1,100) / 100.0   
        leading_slope = (highest_vertex_height) / (self.upper_surface_x[0][highest_vertex])
        trailing_slope = (0.01 - highest_vertex_height) / (1.0 - self.upper_surface_x[0][highest_vertex])
        for i in range(self.n_panels_per_surface - 1):
            curr_vertex = i + 1
            # Highest vertex condition
            if curr_vertex == highest_vertex:
                upper_surface_y[0][curr_vertex] = highest_vertex_height
            # Trailing highest vertex
            elif curr_vertex < highest_vertex:
                x_distance_from_te = self.upper_surface_x[0][curr_vertex] - 1.0
                required_height = x_distance_from_te * trailing_slope + 0.01
                upper_surface_y[0][curr_vertex] = required_height
            # Leading highest vertex    
            else:
                required_height = self.upper_surface_x[0][curr_vertex] * leading_slope
                upper_surface_y[0][curr_vertex] = required_height
        upper_surface_normal = self.get_normal(self.upper_surface_x, upper_surface_y)
        
        # Create lower surface that is legal
        self.lower_surface_x = np.linspace(0,1,self.n_panels_per_surface+1).reshape(1, self.n_panels_per_surface+1)
        lower_surface_y = np.zeros((1,self.n_panels_per_surface+1))
        lower_surface_y[0][-1] = -0.01
        lowest_vertex = np.random.randint(1,self.n_panels_per_surface)
        lowest_vertex_height = -1.0 * np.random.randint(1,100) / 100.0       
        leading_slope = (0.0 - lowest_vertex_height) / (0.0 - self.lower_surface_x[0][lowest_vertex])
        trailing_slope = (-0.01 - lowest_vertex_height) / (1.0 - self.lower_surface_x[0][lowest_vertex])       
        for i in range(self.n_panels_per_surface - 1):
            curr_vertex = i + 1
            # Highest vertex condition
            if curr_vertex == lowest_vertex:
                lower_surface_y[0][curr_vertex] = lowest_vertex_height
            # Trailing highest vertex
            elif curr_vertex > lowest_vertex:
                x_distance_from_te = self.lower_surface_x[0][curr_vertex] - 1.0
                required_height = x_distance_from_te * trailing_slope - 0.01
                lower_surface_y[0][curr_vertex] = required_height
            # Leading highest vertex    
            else:
                required_height = (self.lower_surface_x[0][curr_vertex] * leading_slope)
                lower_surface_y[0][curr_vertex] = required_height
        lower_surface_normal = self.get_normal(self.lower_surface_x, lower_surface_y)
     
        # Combine upper and lower surfaces
        self.surface_x = np.append(self.upper_surface_x[:,:-1], self.lower_surface_x).reshape(1, 2 * n_panels_per_surface + 1)
        self.surface_y = np.append(upper_surface_y[:,:-1], lower_surface_y).reshape(1, 2 * n_panels_per_surface + 1)
        self.surface_normal = np.append(upper_surface_normal, lower_surface_normal, axis=1)
     
    # Gets the normal vectors of the panels of either to upper or lower surface
    # @param x - x coordinates of the panel vertices
    # @param y - y coordinates of the panel vertices
    # @return the normal vertors for each panel
    def get_normal(self, x, y):
        panels = np.array([(x - np.roll(x, -1))[0][:-1], (y - np.roll(y, -1))[0][:-1], np.zeros(self.n_panels_per_surface)])
        product = np.cross(self.z_dirn, panels, axis=0)
        product_norm = np.linalg.norm(product,axis=0)
        return (product / product_norm)
    
    # Solves the integral required to populate linear system to solve for gamma for each panel (positive circulation into page)
    # @param pj - jth panel point written in 3D coords np.array([[x],[y],[z]])
    # @param pjp1 - j+1th panel point written in 3D coords np.array([[x],[y],[z]])
    # @param cp - control point written in 3D coords np.array([[x],[y],[z]])
    # @param cp_normal - the normal vector of the control point in form np.array([[nx],[ny],[0]])
    # @return integral in form vn_prime_j, vn_prime_jp1,
    def solve_integral(self, pj, pjp1, cp, cp_normal):
        # Panel parameters
        panel_length = np.linalg.norm(pjp1 - pj)
        
        # Parameters used for integration
        s = np.array([np.linspace(pj[0],pjp1[0],self.precision).reshape(self.precision), 
                      np.linspace(pj[1],pjp1[1],self.precision).reshape(self.precision), 
                      np.linspace(0,0,self.precision)])
        s_norm = np.linspace(0,panel_length,self.precision).reshape(self.precision)
        
        # Calculate the radius to control point and its square norm
        r = cp - s
        r_norm_sq = np.einsum('ij,ij->j', r, r).reshape(1,self.precision)
        
        # Setup integrals
        den = 2 * np.pi * r_norm_sq
        num_j = np.linspace(1,0,self.precision)
        num_jp1 = np.linspace(0,1,self.precision)
        _rx = -1.0 * r[0].reshape(1,self.precision)
        ry = r[1].reshape(1,self.precision)
        
        # solve integrals
        vx_prime_j = np.trapz(ry * num_j / den, x=s_norm).item()
        vx_prime_jp1 = np.trapz(ry * num_jp1 / den, x=s_norm).item()
        vy_prime_j = np.trapz(_rx * num_j / den, x=s_norm).item()
        vy_prime_jp1 = np.trapz(_rx * num_jp1 / den, x=s_norm).item()

        #format normal vector
        nx = cp_normal[0][0]
        ny = cp_normal[1][0]
        
        # Format outpout
        vn_prime_j = nx * vx_prime_j + ny * vy_prime_j
        vn_prime_jp1 = nx * vx_prime_jp1 + ny * vy_prime_jp1
        
        # Combine results
        return vn_prime_j, vn_prime_jp1
    
    #Solves for the A matrix
    # @return the A matrix
    def get_A(self):
        
        # Init the A matrix
        A = np.zeros((2 * self.n_panels_per_surface + 1, 2 * self.n_panels_per_surface + 1))
        
        # Step through all panels
        for curr_control_point in range(2 * self.n_panels_per_surface):
            
            # Get the control point on the current panel (spatial average of panel boudning points)
            control_point_x = (self.surface_x[0][curr_control_point] + self.surface_x[0][curr_control_point+1])/2
            control_point_y = (self.surface_y[0][curr_control_point] + self.surface_y[0][curr_control_point+1])/2
            control_point = np.array([[control_point_x],[control_point_y],[0.0]])
            
            # Gather the normal vector
            control_point_normal = self.surface_normal[:,curr_control_point].reshape(3,1)
            
            # Step through all inducing panels
            for curr_inducing_panel in range(2 * self.n_panels_per_surface):
                
                # Get the bounding points of the inducing panel
                pj = np.array([[self.surface_x[0][curr_inducing_panel]], [self.surface_y[0][curr_inducing_panel]], [0.0]])
                pjp1 = np.array([[self.surface_x[0][curr_inducing_panel+1]], [self.surface_y[0][curr_inducing_panel+1]], [0.0]])
                
                # Solve the integral
                vn_prime_j, vn_prime_jp1 = self.solve_integral(pj, pjp1, control_point, control_point_normal)
                
                # Format and update A
                A[curr_control_point][curr_inducing_panel] += vn_prime_j
                A[curr_control_point][curr_inducing_panel + 1] += vn_prime_jp1
                
        # Apply the kutta condition
        A[2 * self.n_panels_per_surface][0] = 1.0
        A[2 * self.n_panels_per_surface][2 * self.n_panels_per_surface] = 1.0
        
        return A
            
        
    # Solves for the B matrix 
    # @param V_inf - freestream velocity in form np.array([[Vx],[Vy],[0]])
    # @return B matrix
    def get_B(self, v_inf):
        
        # init and populate B
        B = np.matmul(v_inf.reshape(1,3), self.surface_normal).reshape(2 * self.n_panels_per_surface,1)
        
        # Kutta condition
        B = np.append(B, 0.0).reshape(2 * self.n_panels_per_surface + 1, 1)
        
        return B
    
    # Solves for the circulation strength at each panel vertex
    # @param V_inf - the free stream velocity
    # @return the circulation strength at each panel vertex
    def solve_gamma(self, v_inf):
        A = self.get_A()
        B = self.get_B(v_inf)
        return np.linalg.solve(A, -1.0 * B)
    
    # Solves for the local tangential velcoity induced by panel j at control point cp
    # @param pj - jth panel point written in 3D coords np.array([[x],[y],[z]])
    # @param pjp1 - j+1th panel point written in 3D coords np.array([[x],[y],[z]])
    # @param cp - control point written in 3D coords np.array([[x],[y],[z]])
    # @param gamma_j - the circulation at point pj
    # @param gamma_jp1 - the circulation at point pjp1
    # @return velocity at cp in form v_induced
    def solve_velocity(self, pj, pjp1, cp, gamma_j, gamma_jp1):
        # Panel parameters
        panel_length = np.linalg.norm(pjp1 - pj)
        
        # Parameters used for integration
        s = np.array([np.linspace(pj[0],pjp1[0],self.precision).reshape(self.precision), 
                      np.linspace(pj[1],pjp1[1],self.precision).reshape(self.precision), 
                      np.linspace(0,0,self.precision)])
        s_norm = np.linspace(0,panel_length,self.precision).reshape(self.precision)
        
        # Calculate the radius to control point and its square norm
        r = cp - s
        r_norm_sq = np.einsum('ij,ij->j', r, r).reshape(1,self.precision)
        
        # Setup integrals
        den = 2 * np.pi * r_norm_sq
        num_j = np.linspace(1,0,self.precision)
        num_jp1 = np.linspace(0,1,self.precision)
        _rx = -1.0 * r[0].reshape(1,self.precision)
        ry = r[1].reshape(1,self.precision)
        
        # solve integrals
        vx_prime_j = np.trapz(ry * num_j / den, x=s_norm).item()
        vx_prime_jp1 = np.trapz(ry * num_jp1 / den, x=s_norm).item()
        vy_prime_j = np.trapz(_rx * num_j / den, x=s_norm).item()
        vy_prime_jp1 = np.trapz(_rx * num_jp1 / den, x=s_norm).item()
        
        # Format outpout
        vx = vx_prime_j * gamma_j + vx_prime_jp1 * gamma_jp1
        vy = vy_prime_j * gamma_j + vy_prime_jp1 * gamma_jp1
        vz = 0.0

        # Combine results
        return np.array([vx, vy, vz])
    
    # Gets the velocity induced by the freestream and the vortex panels at each control point
    # @param V_inf - freestream velocity in form np.array([[Vx],[Vy],[0]])
    # @return coefficient of pressure at each control point
    def solve_cp(self, v_inf):
        gamma = self.solve_gamma(v_inf)
        
        # Init the induced velocity vector
        v_induced = np.zeros((2 * self.n_panels_per_surface, 3))
        
        # Step through all panels
        for curr_control_point in range(2 * self.n_panels_per_surface):
            
            # Get the control point on the current panel (spatial average of boudning points)
            control_point_x = (self.surface_x[0][curr_control_point] + self.surface_x[0][curr_control_point+1])/2
            control_point_y = (self.surface_y[0][curr_control_point] + self.surface_y[0][curr_control_point+1])/2
            control_point = np.array([[control_point_x],[control_point_y],[0.0]])
            
            # Step through all inducing panels
            for curr_inducing_panel in range(2 * self.n_panels_per_surface):
                
                # Get the bounding points of the inducing panel
                pj = np.array([[self.surface_x[0][curr_inducing_panel]], [self.surface_y[0][curr_inducing_panel]], [0.0]])
                pjp1 = np.array([[self.surface_x[0][curr_inducing_panel+1]], [self.surface_y[0][curr_inducing_panel+1]], [0.0]])
                gamma_j = gamma[curr_inducing_panel][0]
                gamma_jp1 = gamma[curr_inducing_panel + 1][0]
                
                # Solve the integral
                v_induced[curr_control_point] += self.solve_velocity(pj, pjp1, control_point, gamma_j, gamma_jp1)
        
        # Calculate tanjential velocity at every control point
        v = v_induced[:] + v_inf.reshape(3)
        v_mag = np.linalg.norm(v,axis=1).reshape(self.n_panels_per_surface * 2, 1)
    
        # Bernoulli's equation to get Cp
        v_inf_mag = np.linalg.norm(v_inf)
        cp = 1 - ((v_mag ** 2) / (v_inf_mag ** 2))
        return cp
    
    # Gets the axial, normal, and moment coefficients based on the pressure distribution
    # @param V_inf - freestream velocity in form np.array([[Vx],[Vy],[0]])
    # @return the axial, normal, and moment coefficients
    def solve_ca_cn_cm4c(self, v_inf):
        # Get and split cp into lower and upper
        cp = self.solve_cp(v_inf)
        cp_upper = cp[0:self.n_panels_per_surface][::-1].reshape(self.n_panels_per_surface)
        cp_lower = cp[self.n_panels_per_surface:2*self.n_panels_per_surface].reshape(self.n_panels_per_surface)
        
        # Get and split x/c coords
        x_coords = ((self.surface_x + np.roll(self.surface_x,-1)) / 2)[0][self.n_panels_per_surface:2*self.n_panels_per_surface]

        # Get and split y/c coords
        y_coords_upper = ((self.surface_y + np.roll(self.surface_y,-1)) / 2)[0][0:self.n_panels_per_surface][::-1]
        y_coords_lower = ((self.surface_y + np.roll(self.surface_y,-1)) / 2)[0][self.n_panels_per_surface:2*self.n_panels_per_surface]
        
        # Solve for differential slopes
        x_verts_upper = (self.surface_x[0][0:self.n_panels_per_surface+1])[::-1]
        x_verts_lower = (self.surface_x[0][self.n_panels_per_surface:2*self.n_panels_per_surface+1])
        y_verts_upper = (self.surface_y[0][0:self.n_panels_per_surface+1])[::-1]
        y_verts_lower = (self.surface_y[0][self.n_panels_per_surface:2*self.n_panels_per_surface+1])
        dy_upper = (np.roll(y_verts_upper,-1) - y_verts_upper)[:-1]
        dy_lower = (np.roll(y_verts_lower,-1) - y_verts_lower)[:-1]
        dx_upper = (np.roll(x_verts_upper,-1) - x_verts_upper)[:-1]
        dx_lower = (np.roll(x_verts_lower,-1) - x_verts_lower)[:-1]
        slope_upper = dy_upper / dx_upper
        slope_lower = dy_lower / dx_lower
        
        # Solve for ca and cn
        ca = np.trapz(cp_upper*slope_upper - cp_lower*slope_lower, x=x_coords)
        cn = np.trapz(cp_lower - cp_upper, x=x_coords)
        
        # Solve for cm4c
        part_1 = np.trapz((cp_upper - cp_lower) * (0.25 - x_coords), x=x_coords)
        part_2 = np.trapz((cp_upper*slope_upper*y_coords_upper) - (cp_lower*slope_lower*y_coords_lower), x=x_coords)
        cm4c = part_1 + part_2
        
        return ca, cn, cm4c

    # Gets the lift, pressure drag, and moment coefficients based on the pressure distribution
    # @param v_inf - freestream velocity in form np.array([[Vx],[Vy],[0]])
    # @return the lift, pressure drag, and moment coefficients coefficient
    def solve_cl_cdp_cm4c(self, v_inf):
        ca, cn, cm4c = self.solve_ca_cn_cm4c(v_inf)
        alpha = np.arctan(v_inf[1][0] / v_inf[0][0])
        
        # Get lift and drag coeffs
        cl = cn * np.cos(alpha) - ca * np.sin(alpha)
        cdp = cn * np.sin(alpha) + ca * np.cos(alpha)
        
        return cl, cdp, cm4c

    # converts discrete action, a, into an airfoil transforming action
    # @param a - the action index to be converted. Given n_panels_per_surface, the action space is 4 * n_panels_per_surface - 2
    # @return the airfoil transforming action set in form np.array(2 * n_panels_per_surface,)
    def a_to_action(self,a):
        # Action set can either move a vertex up by 10% or down by 10%
        # There are 2*n_panels_per_surface + 1 vertices, with 2*n_panels_per_surface alterable vertices
        # Therefore we must limit an airfoil transforming action set to alter only one vertex at a time
        multiplier = 0.9 * (1 - a % 2) + 1.1 * (a % 2)
        vertex = a // 2
        if vertex >= self.n_panels_per_surface:
            vertex += 1
            
        action = np.ones(2 * self.n_panels_per_surface)
        action[vertex] = multiplier
            
        return action

    # Performs the action on the airfoil and returns a reward
    # @param a - the action index
    # @param vis_foil=False - determined whether the new airfoil is saved as an image
    # @param n - number label of airfoil
    # @return the next state, reward, and whether to terminate simulator
    def step(self, a, vis_foil=False, n=0):
        
        # Get the airfoil transforming action set
        action = self.a_to_action(a)
        
        # Determine if simulation is complete
        self.curr_step += 1
        done = (self.curr_step == self.max_num_steps)
        
        # Perform the action set on the state
        s1 = self.surface_y
        temp = self.surface_y[0][:-1] * action
        s2 = np.append(temp, temp[0]-0.02).reshape(1,2*self.n_panels_per_surface+1)
            
        # Determine airfoil spikeness
        y_coords_upper = (s2[0][0:self.n_panels_per_surface+1])[::-1]
        y_coords_lower = (s2[0][self.n_panels_per_surface:2*self.n_panels_per_surface+1])
        n_peaks_upper = np.size(find_peaks(y_coords_upper)[0])
        n_peaks_lower = np.size(find_peaks(-1.0*y_coords_lower)[0])
        
        # Determine max flow turning angle on upper and lower surfaces
        
        
        ####################################################### REWARD FUNCTION #######################################################
        # If the action moves any points outside of the acceptable range, return a large negative reward and the old airfoil
        # The acceptable range is any y/c between [-1.0, 1.0]
        # The acceptable range for the TE is y/c between [-0.10,0.10]
        if (max(s2[0]) > 1.0 or min(s2[0]) < -1.0) or (s2[0][0] > 0.10 or s2[0][-1] < -0.10):
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n)
            return s1, -100.0, done
        
        # If the lower surface every intersects the upper surface anywhere but the LE, return a large negative reward and the new airfoil
        elif (y_coords_upper < y_coords_lower)[1:].any():
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n)
            return s1, -100.0, done
        
        # If the airfoil is too spikey, return a negative reward and the new airfoil
        # Too spikey is defined as having more than a 2 peaks per surface
        elif ((n_peaks_upper > 2) or (n_peaks_lower > 2)):
            # Update the stored airfoil
            upper_surface_y = s2[0][0:self.n_panels_per_surface+1].reshape(1, self.n_panels_per_surface+1)
            lower_surface_y = s2[0][self.n_panels_per_surface: 2*self.n_panels_per_surface+1].reshape(1, self.n_panels_per_surface+1)
            upper_surface_normal = self.get_normal(self.upper_surface_x, upper_surface_y)
            lower_surface_normal = self.get_normal(self.lower_surface_x, lower_surface_y)
            self.surface_y = s2
            self.surface_normal = np.append(upper_surface_normal, lower_surface_normal, axis=1)
        
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n)
            return s2.reshape(2 * self.n_panels_per_surface + 1), -10.0, done
        
        # If the airfoil has a turning angle that is too great (>90 degrees), return a negative reward and the new airfoil
        elif (False):
            return s2.reshape(2 * self.n_panels_per_surface + 1), -10.0, done
        
        # If the action is acceptable, return a reward proportional to the mean abs percent error between the new airfoil and the design parameters
        else:
            # Update the stored airfoil
            upper_surface_y = s2[0][0:self.n_panels_per_surface+1].reshape(1, self.n_panels_per_surface+1)
            lower_surface_y = s2[0][self.n_panels_per_surface: 2*self.n_panels_per_surface+1].reshape(1, self.n_panels_per_surface+1)
            upper_surface_normal = self.get_normal(self.upper_surface_x, upper_surface_y)
            lower_surface_normal = self.get_normal(self.lower_surface_x, lower_surface_y)
            self.surface_y = s2
            self.surface_normal = np.append(upper_surface_normal, lower_surface_normal, axis=1)
            
            # init loss sum to 0.0
            cl_loss = 0.0
            cdp_loss = 0.0
            cm4c_loss = 0.0
            
            n_test_points = np.size(self.v_inf_test_points)
            for test_point in range(n_test_points):
                
                # Get the current velocity vector
                v_inf = np.array([[self.v_inf_test_points[test_point] * np.cos(self.alpha_test_points[test_point])], 
                                  [self.v_inf_test_points[test_point] * np.sin(self.alpha_test_points[test_point])], 
                                  [0.0]])
                
                # Use the vortex panel method to solve for the airfoil's non-dimensional parameters
                cl, cdp, cm4c = self.solve_cl_cdp_cm4c(v_inf)
                
                # Update the loss function
                cl_loss += abs((cl - self.cl_test_points[test_point]) / self.cl_test_points[test_point])
                cdp_loss += abs((cdp - self.cdp_test_points[test_point]) / self.cdp_test_points[test_point])
                cm4c_loss += abs((cm4c - self.cm4c_test_points[test_point]) / self.cm4c_test_points[test_point])
                
            # Calculate the total weighted loss
            # Adjust weights to get more tuned results
            cl_loss = cl_loss / n_test_points
            cdp_loss = cdp_loss / n_test_points
            cm4c_loss = cm4c_loss / n_test_points
            cl_loss_weight = 5.0
            cdp_loss_weight = 2.0
            cm4c_loss_weight = 1.0
            total_loss = (cl_loss_weight*cl_loss + cdp_loss_weight*cdp_loss + cm4c_loss_weight*cm4c_loss)/(cl_loss_weight + cdp_loss_weight + cm4c_loss_weight)
        
            # Use the loss to get a reward
            # The size of this clip determines the size of the reward return space
            reward = 5 - np.clip(total_loss, 0.0,5.0)
            
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n)
            return s2.reshape(2 * self.n_panels_per_surface + 1), reward, done
        
        ####################################################### /REWARD FUNCTION ######################################################
        
    # Resets the environment to a random airfoil and returns the new airfoil
    # @param vis_foil=False - determined whether the new airfoil is saved as an image
    # @param n=0 - the number label of the newairfoil
    # @return the new airfoil y/c coords (x/c not returned due to even spacing)
    def reset(self, vis_foil=False, n=0):
        self.curr_step = 0
        
        # Create upper surface that is legal
        upper_surface_y = np.zeros((1,self.n_panels_per_surface+1))
        upper_surface_y[0][0] = 0.01
        highest_vertex = np.random.randint(1,self.n_panels_per_surface)
        highest_vertex_height = np.random.randint(1,100) / 100.0   
        leading_slope = (highest_vertex_height) / (self.upper_surface_x[0][highest_vertex])
        trailing_slope = (0.01 - highest_vertex_height) / (1.0 - self.upper_surface_x[0][highest_vertex])
        for i in range(self.n_panels_per_surface - 1):
            curr_vertex = i + 1
            # Highest vertex condition
            if curr_vertex == highest_vertex:
                upper_surface_y[0][curr_vertex] = highest_vertex_height
            # Trailing highest vertex
            elif curr_vertex < highest_vertex:
                x_distance_from_te = self.upper_surface_x[0][curr_vertex] - 1.0
                required_height = x_distance_from_te * trailing_slope + 0.01
                upper_surface_y[0][curr_vertex] = required_height
            # Leading highest vertex    
            else:
                required_height = self.upper_surface_x[0][curr_vertex] * leading_slope
                upper_surface_y[0][curr_vertex] = required_height
        upper_surface_normal = self.get_normal(self.upper_surface_x, upper_surface_y)
        
        # Create lower surface that is legal
        lower_surface_y = np.zeros((1,self.n_panels_per_surface+1))
        lower_surface_y[0][-1] = -0.01
        lowest_vertex = np.random.randint(1,self.n_panels_per_surface)
        lowest_vertex_height = -1.0 * np.random.randint(1,100) / 100.0       
        leading_slope = (0.0 - lowest_vertex_height) / (0.0 - self.lower_surface_x[0][lowest_vertex])
        trailing_slope = (-0.01 - lowest_vertex_height) / (1.0 - self.lower_surface_x[0][lowest_vertex])       
        for i in range(self.n_panels_per_surface - 1):
            curr_vertex = i + 1
            # Highest vertex condition
            if curr_vertex == lowest_vertex:
                lower_surface_y[0][curr_vertex] = lowest_vertex_height
            # Trailing highest vertex
            elif curr_vertex > lowest_vertex:
                x_distance_from_te = self.lower_surface_x[0][curr_vertex] - 1.0
                required_height = x_distance_from_te * trailing_slope - 0.01
                lower_surface_y[0][curr_vertex] = required_height
            # Leading highest vertex    
            else:
                required_height = (self.lower_surface_x[0][curr_vertex] * leading_slope)
                lower_surface_y[0][curr_vertex] = required_height
        lower_surface_normal = self.get_normal(self.lower_surface_x, lower_surface_y)
     
        # Combine upper and lower surfaces
        self.surface_x = np.append(self.upper_surface_x[:,:-1], self.lower_surface_x).reshape(1, 2 * self.n_panels_per_surface + 1)
        self.surface_y = np.append(upper_surface_y[:,:-1], lower_surface_y).reshape(1, 2 * self.n_panels_per_surface + 1)
        self.surface_normal = np.append(upper_surface_normal, lower_surface_normal, axis=1)
        
        # Visualize airfoil
        if(vis_foil):
            self.visualize_airfoil(n)
        
        return self.surface_y.reshape(2 * self.n_panels_per_surface + 1)
    
    # Visualizes the pressure distribution over the airfoil
    # @param cp - pressure distribution of airfoil
    # @param n - number label of airfoil
    def visualize_cp(self, cp, n):
        x_coords = ((self.surface_x + np.roll(self.surface_x,-1)) / 2)[0][:-1].reshape(2*self.n_panels_per_surface)
        plt.clf
        plt.scatter(x_coords, cp.reshape(2*self.n_panels_per_surface))
        plt.plot(x_coords, cp.reshape(2*self.n_panels_per_surface))
        plt.gca().invert_yaxis()
        title_str = "Cp for Airfoil " + str(n)
        plt.title(title_str)
        plt.xlabel("x/c [unitless]")
        plt.ylabel("Cp [unitless]")
        plt.xlim([0,1])
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        save_str = "cp_airfoil_" + str(n) + ".png"
        plt.savefig(save_str, dpi = 100)
        plt.close()        
    
    # Visualizes the shape of the airfoil
    # @param n - number label of airfoil
    def visualize_airfoil(self, n):
        plt.clf
        plt.scatter(self.surface_x.reshape(2*self.n_panels_per_surface + 1), self.surface_y.reshape(2*self.n_panels_per_surface + 1))
        plt.plot(self.surface_x.reshape(2*self.n_panels_per_surface + 1), self.surface_y.reshape(2*self.n_panels_per_surface + 1))
        title_str = "Airfoil " + str(n)
        plt.title(title_str)
        plt.xlabel("x/c [unitless]")
        plt.ylabel("y/c [unitless]")
        plt.xlim([0,1])
        plt.ylim([-1,1])
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        save_str = "airfoils/airfoil_" + str(n) + ".png"
        plt.savefig(save_str, dpi = 100)
        plt.close()