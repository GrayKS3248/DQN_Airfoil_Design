"""
Created on Mon Sep 28 14:23:01 2020

@author: Grayson Schaer
"""
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import csv

# Vortex panel method solver and environment stepper for a 1m chord length airfoil
# @param max_num_steps - the max number of times an airfoil can be changed by an agent
# @param n_panels_per_surface - the number of panels per surface on the airfoil
# @param alpha_test_points - n angles of attack in form np.array([a0, a1, a2, a3, a4, ..., an])
# @param cl_test_points - n lift coefficients in form np.array([cl0, ..., cln])
# @param cdp_test_points - n pressure drag coefficients in form np.array([cdp0, ..., cdpn])
# @param cm4c_test_points - n moment coefficients about quater chord in form np.array([cm4c0, ..., cm4cn])
class Vortex_Panel_Solver():
    def __init__(self, max_num_steps, n_panels_per_surface, re_test, alpha_test_points, cl_test_points, cdp_test_points, cm4c_test_points, symmetric=False, debug=False):
        
        self.max_num_steps = max_num_steps
        self.curr_step = 0
        self.num_actions = 4 * n_panels_per_surface - 5
        self.state_dimension = 2 * n_panels_per_surface + 1
        self.n_panels_per_surface = n_panels_per_surface
        self.precision = 14 # ***** MUST BE EVEN ***** #
        
        self.p_inf = 101325.0
        self.rho = 1.225
        self.mu = 1.849e-5
        self.v_free = (re_test * self.mu) / (self.rho)
        self.q_inf = 0.5 * self.rho * self.v_free**2
        
        self.alpha_test_points = alpha_test_points
        self.cl_test_points = cl_test_points
        self.cdp_test_points = cdp_test_points
        self.cm4c_test_points = cm4c_test_points
        
        self.num_j = np.reshape(np.linspace(1,0,self.precision),(self.precision,1))
        self.num_jp1 = np.reshape(np.linspace(0,1,self.precision),(self.precision,1))
        
        # Create upper surface that is legal
        upper_surface_normal, upper_surface_y = self.make_upper_surface(symmetric)
        
        # Create lower surface that is legal
        lower_surface_normal, lower_surface_y = self.make_lower_surface(symmetric)
     
        # Combine upper and lower surfaces
        self.surface_x = np.append(self.upper_surface_x[:-1], self.lower_surface_x)
        self.surface_y = np.append(upper_surface_y[:-1], lower_surface_y)
        self.surface_normal = np.append(upper_surface_normal, lower_surface_normal, axis=1)
        self.x_cen_panel = ((self.surface_x + np.roll(self.surface_x,-1)) / 2)[self.n_panels_per_surface:2*self.n_panels_per_surface]
        
        # Debug mode
        if debug:
            self.visualize_airfoil(0,'debug/')
            cl, cdp, cm4c, ok = self.solve_cl_cdp_cm4c(np.array([1.0, 0.0]))
            assert(abs(cl) <= 1e-13)
            assert(abs(cm4c) <= 1e-13)
            print("cl delta: " + str(cl))
            print("cm4c delta: " + str(cm4c))
            print("Unit Test passed!")
     
    # Create upper surface that is legal
    def make_upper_surface(self, symmetric):
        self.upper_surface_x = np.append(np.array([1.0]), ((np.linspace(0.97468,0.0,self.n_panels_per_surface))**2))
        upper_surface_y = np.zeros(self.n_panels_per_surface+1)
        upper_surface_y[0] = 0.01
        
        if not (symmetric):
            highest_vertex = np.random.randint(self.n_panels_per_surface - self.n_panels_per_surface // 2, self.n_panels_per_surface)
            highest_vertex_height = np.random.randint(25,83) / 666.66667 
        else:
            highest_vertex = self.n_panels_per_surface - self.n_panels_per_surface // 2
            highest_vertex_height = 0.10
            
        leading_slope = (highest_vertex_height) / (self.upper_surface_x[highest_vertex])
        trailing_slope = (0.01 - highest_vertex_height) / (1.0 - self.upper_surface_x[highest_vertex])
        
        for i in range(self.n_panels_per_surface - 1):
            curr_vertex = i + 1
            # Highest vertex condition
            if curr_vertex == highest_vertex:
                upper_surface_y[curr_vertex] = highest_vertex_height
            # Trailing highest vertex
            elif curr_vertex < highest_vertex:
                x_distance_from_te = self.upper_surface_x[curr_vertex] - 1.0
                required_height = x_distance_from_te * trailing_slope + 0.01
                upper_surface_y[curr_vertex] = required_height
            # Leading highest vertex    
            else:
                required_height = self.upper_surface_x[curr_vertex] * leading_slope
                upper_surface_y[curr_vertex] = required_height
                
        upper_surface_normal = self.get_normal(self.upper_surface_x, upper_surface_y)
        
        return upper_surface_normal, upper_surface_y
     
    # Create upper surface that is legal
    def make_lower_surface(self, symmetric):
        self.lower_surface_x = np.append(((np.linspace(0.0,0.97468,self.n_panels_per_surface))**2), np.array([1.0]))
        lower_surface_y = np.zeros(self.n_panels_per_surface+1)
        lower_surface_y[-1] = -0.01
        
        if not (symmetric):
            lowest_vertex = np.random.randint(1,self.n_panels_per_surface // 2 + 3)
            lowest_vertex_height = -1.0 * np.random.randint(25,83) / 666.66667    
        else:
            lowest_vertex = self.n_panels_per_surface // 2
            lowest_vertex_height = -0.10
            
        leading_slope = (0.0 - lowest_vertex_height) / (0.0 - self.lower_surface_x[lowest_vertex])
        trailing_slope = (-0.01 - lowest_vertex_height) / (1.0 - self.lower_surface_x[lowest_vertex])   
        
        for i in range(self.n_panels_per_surface - 1):
            curr_vertex = i + 1
            # Highest vertex condition
            if curr_vertex == lowest_vertex:
                lower_surface_y[curr_vertex] = lowest_vertex_height
            # Trailing highest vertex
            elif curr_vertex > lowest_vertex:
                x_distance_from_te = self.lower_surface_x[curr_vertex] - 1.0
                required_height = x_distance_from_te * trailing_slope - 0.01
                lower_surface_y[curr_vertex] = required_height
            # Leading highest vertex    
            else:
                required_height = (self.lower_surface_x[curr_vertex] * leading_slope)
                lower_surface_y[curr_vertex] = required_height
                
        lower_surface_normal = self.get_normal(self.lower_surface_x, lower_surface_y)
        
        return lower_surface_normal, lower_surface_y
    
    # Gets the normal vectors of the panels of either to upper or lower surface
    # @param x - x coordinates of the panel vertices
    # @param y - y coordinates of the panel vertices
    # @return the normal vertors for each panel
    def get_normal(self, x, y):
        panels = np.array([(x - np.roll(x, -1))[:-1], (y - np.roll(y, -1))[:-1], np.zeros(self.n_panels_per_surface)])
        product = np.array([-1.0*panels[1,:], panels[0,:]])
        product_norm = np.linalg.norm(product,axis=0)
        norm_dirn = product / product_norm
        
        return norm_dirn
    
    # Solves the integral required to populate linear system to solve for gamma for each panel (positive circulation into page)
    # @param pj - panel points set
    # @param pjp1 - j+1th panel points set
    # @param cp - control point written in 2D coords np.array([[[x], [y]]])
    # @param cp_normal - the normal vector of the control point in form np.array([x, y])
    # @return integral in form vn_prime_j, vn_prime_jp1, vx_prime_jp1, vy_prime_j, vy_prime_jp1
    def solve_integral(self, pj_set, pjp1_set, cp, cp_normal):
        # Panel parameters
        panel_length = np.linalg.norm(pj_set - pjp1_set,axis=0)
        
        # Parameters used for integration
        s1 = np.linspace(pj_set,pjp1_set,self.precision,axis=0)
        s_norm = np.linspace(0.0, panel_length, self.precision)
        
        # Calculate the radius to control point and its square norm
        r1 = cp - s1
        r_norm_sq = np.sum(r1**2,axis=1)
        
        # Setup integrals
        den = 2 * np.pi * r_norm_sq
        minus_rx = -1.0 * r1[:,0,:]
        ry = r1[:,1,:]
        
        # solve integrals
        vx_prime_j = np.trapz(ry * (self.num_j / den), x=s_norm, axis=0)
        vx_prime_jp1 = np.trapz(ry * (self.num_jp1 / den), x=s_norm, axis=0)
        vy_prime_j = np.trapz(minus_rx * (self.num_j / den), x=s_norm, axis=0)
        vy_prime_jp1 = np.trapz(minus_rx * (self.num_jp1 / den), x=s_norm, axis=0)

        #format normal vector
        nx = cp_normal[0]
        ny = cp_normal[1]
        
        # Calculate normal velocities
        vn_prime_j = nx * vx_prime_j + ny * vy_prime_j
        vn_prime_jp1 = nx * vx_prime_jp1 + ny * vy_prime_jp1
        
        # Format output
        vn_prime = np.append(vn_prime_j, 0.0) + np.insert(vn_prime_jp1,0,0.0)
        vx_prime = np.append(vx_prime_j, 0.0) + np.insert(vx_prime_jp1,0,0.0)
        vy_prime = np.append(vy_prime_j, 0.0) + np.insert(vy_prime_jp1,0,0.0)
        
        # Combine results
        return vn_prime, vx_prime, vy_prime, panel_length
            
    #Solves for the A matrices
    # @return the A matrix that solves the normal vel mag, the A matrix that solves the vx vel mag, and the A matrix that solves the vy vel mag
    def get_A(self):
        
        # Init the A matrix
        A = np.zeros((2 * self.n_panels_per_surface + 1, 2 * self.n_panels_per_surface + 1))
        A_vx = np.zeros((2 * self.n_panels_per_surface, 2 * self.n_panels_per_surface + 1))
        A_vy = np.zeros((2 * self.n_panels_per_surface, 2 * self.n_panels_per_surface + 1))
        
        # Get the panel vertices
        pj_set = np.array([self.surface_x[:-1], 
                           self.surface_y[:-1]])
        pjp1_set = np.array([np.roll(self.surface_x,-1)[:-1], 
                             np.roll(self.surface_y,-1)[:-1]])
        
        # Get the control points
        control_point_set = (pj_set + pjp1_set) / 2
        
        # Step through all panels
        for curr_control_point in range(2 * self.n_panels_per_surface):
            
            # Get the control point and its normal on the current panel
            control_point = np.reshape(control_point_set[:,curr_control_point],(1,2,1))
            control_point_normal = self.surface_normal[:,curr_control_point]
            
            # Solve the integral
            vn_prime, vx_prime, vy_prime, panel_length = self.solve_integral(pj_set, pjp1_set, control_point, control_point_normal)
                
            # Format and update A
            A[curr_control_point][:] = vn_prime
            A_vx[curr_control_point][:] = vx_prime
            A_vy[curr_control_point][:] = vy_prime
                
        # Apply the kutta condition
        A[2 * self.n_panels_per_surface][0] = 1.0
        A[2 * self.n_panels_per_surface][2 * self.n_panels_per_surface] = 1.0
        
        return A, A_vx, A_vy, panel_length, control_point_set
            
        
    # Solves for the B matrix 
    # @param V_inf - freestream velocity in form np.array([[Vx],[Vy],[0]])
    # @return B matrix
    def get_B(self, v_inf):
        
        # init and populate B
        B = np.matmul(v_inf.reshape(1,2), self.surface_normal).reshape(2 * self.n_panels_per_surface,1)
        
        # Kutta condition
        B = np.append(B, 0.0).reshape(2 * self.n_panels_per_surface + 1, 1)
        
        return B
    
    # Gets the velocity induced by the freestream and the vortex panels at each control point
    # @param V_inf - freestream velocity in form np.array([Vx, Vy])
    # @return coefficient of pressure at each control point
    def solve_cp(self, v_inf):
        # Get the magnitude of the tangential vel at each control point
        A, A_vx, A_vy, panel_length, control_point_set = self.get_A()
        B = self.get_B(v_inf)
        gamma = np.linalg.solve(A, -1.0 * B)
        vx = ((np.matmul(A_vx, gamma)) + v_inf[0]).reshape(2*self.n_panels_per_surface)
        vy = ((np.matmul(A_vy, gamma)) + v_inf[1]).reshape(2*self.n_panels_per_surface)
        v_mag = np.linalg.norm(np.array([vx,vy]), axis=0)
    
        # Bernoulli's equation to get Cp
        cp = 1 - v_mag ** 2
        return cp, panel_length, control_point_set

    # Gets the lift, pressure drag, and moment coefficients based on the pressure distribution
    # @param v_inf - freestream velocity in form np.array([Vx, Vy])
    # @return the lift, pressure drag, and moment coefficients coefficient
    def solve_cl_cdp_cm4c(self, v_inf):
        # Get and split cp into lower and upper
        cp, panel_length, control_point_set = self.solve_cp(v_inf)
        cp_upper = cp[0:self.n_panels_per_surface][::-1]
        cp_lower = cp[self.n_panels_per_surface:2*self.n_panels_per_surface]
        cp_upper_peaks = np.insert(cp_upper, 0, cp_lower[0])
        cp_lower_peaks = np.insert(cp_lower, 0, cp_upper[0])
        
        # Make sure that the pressure distribution on the upper surface does not intersect the pressure distribution on the lower surface
        ok_cp_distribution = (cp_upper < cp_lower).all()
        
        # Make sure that the suction peak is on the first quarter of the airfoil
        ok_suction_peak = control_point_set[0,:][self.n_panels_per_surface+np.argmin(cp_upper)] <= 0.25
        
        # Make sure that the pressure distribution on both surfaces no more than 2 peaks
        n_peaks_upper = len(find_peaks(-1.0*cp_upper_peaks)[0])
        n_peaks_lower = len(find_peaks(-1.0*cp_lower_peaks)[0])
        ok_cp_shape = (n_peaks_upper <= 2 and n_peaks_lower <= 2)
        
        # Calculate normal and axial force coefficients
        TE_correction = np.array([(-self.p_inf*panel_length*self.surface_normal[0,:]).sum(),0.0])
        p_dist = self.q_inf*cp + self.p_inf
        f_dist = -p_dist*panel_length*self.surface_normal
        f_net = f_dist.sum(axis=1) - TE_correction
        coeff = f_net / self.q_inf
        
        # Calculate moment about quarter chord
        torque = np.cross(np.transpose(control_point_set - np.array([[0.25],[0.0]])), np.transpose(f_dist)).sum()
        cm4c = torque / self.q_inf
        
        # Calculate lift and pressure drag coefficients
        cl = coeff[1]*v_inf[0] - coeff[0]*v_inf[1]
        cdp = coeff[1]*v_inf[1] + coeff[0]*v_inf[0]
        
        return cl, cdp, cm4c, (ok_cp_distribution and ok_suction_peak and ok_cp_shape)

    # converts discrete action, a, into an airfoil transforming action
    # @param a - the action index to be converted. Given n_panels_per_surface, the action space is 4 * n_panels_per_surface - 4
    # @return the airfoil transforming action set in form np.array(2 * n_panels_per_surface,)
    def a_to_action(self,a):
        # Action set can either move a vertex up or down by y/c = 0.01
        # There are 2*n_panels_per_surface + 1 vertices, with 2*n_panels_per_surface - 1 alterable vertices (not LE or TE)
        # Therefore we must limit an airfoil transforming action set to alter only one vertex at a time
        if a < 0:
            a = 0
        if a > 4 * self.n_panels_per_surface - 5:
            a = 4 * self.n_panels_per_surface - 5
        adder = -0.01 * (1 - a % 2) + 0.01 * (a % 2)
        vertex = (a // 2) + 1
        if vertex >= self.n_panels_per_surface:
            vertex += 1
            
        action = np.zeros(2 * self.n_panels_per_surface + 1)
        action[vertex] = adder
            
        return action

    # Performs the action on the airfoil and returns a reward
    # @param a - the action index
    # @param vis_foil=False - determined whether the new airfoil is saved as an image
    # @param n - number label of airfoil
    # @param reward_depreciation - the constant by which the reward range is multiplied. Used to degrade reward range over time.
    # @return the next state, reward, and whether to terminate simulator
    def step(self, a, vis_foil=False, n=0, reward_depreciation=1.0, path=""):
        
        # Get the airfoil transforming action set
        action = self.a_to_action(a)
        
        # Determine if simulation is complete
        self.curr_step += 1
        done = (self.curr_step == self.max_num_steps)
        
        # Perform the action set on the state
        s1 = self.surface_y
        s2 = self.surface_y + action
            
        # Determine new airfoil geometry
        upper_surface_y = s2[0:self.n_panels_per_surface+1]
        lower_surface_y = s2[self.n_panels_per_surface: 2*self.n_panels_per_surface+1]
        upper_surface_normal = self.get_normal(self.upper_surface_x, upper_surface_y)
        lower_surface_normal = self.get_normal(self.lower_surface_x, lower_surface_y)
        surface_normal_new = np.append(upper_surface_normal, lower_surface_normal, axis=1)
    
        # Determine airfoil spikeness
        upper_surface_y = np.flip(upper_surface_y)
        
        # Determine max flow turning angle on upper and lower surfaces (excluding LE and TE)
        surface_tan_new = np.array([surface_normal_new[1,:], -1.0 * surface_normal_new[0,:]])
        surface_tan_new_roll = np.roll(surface_tan_new,-1,axis=1)
        LE_index = self.n_panels_per_surface-1
        TE_index = 2*self.n_panels_per_surface-1
        new_turning_angles = np.delete(np.einsum('ij,ij->j', surface_tan_new, surface_tan_new_roll), [LE_index, TE_index])
        
        # Determine the maximum thickness and its location
        maximum_thickness = max(upper_surface_y - lower_surface_y)
        maximum_thickness_location = self.surface_x[np.argmax(upper_surface_y - lower_surface_y) + self.n_panels_per_surface]
        
        # Determine the minimum thickness location
        minimum_thickness = min((upper_surface_y - lower_surface_y)[1:-1])
        
        # Determine the spikiness of the thickness distribution
        n_peaks_thickness = len(find_peaks((upper_surface_y-lower_surface_y))[0])
        
        ####################################################### REWARD FUNCTION #######################################################
        # If the action moves any points outside of the acceptable range, return a negative reward and the old airfoil
        # The acceptable range is any y/c between [-0.5, 0.5]
        # The acceptable range for the TE is y/c between [-0.10,0.10]
        if (max(s2) > 0.5 or min(s2) < -0.5) or (s2[0] > 0.10 or s2[-1] < -0.10):
            #print("out of bounds")
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n, path=path)
            return s1, -1.0, done
        
        # If the lower surface every intersects the upper surface anywhere but the LE, return a negative reward and the new airfoil
        elif (upper_surface_y < lower_surface_y).any():
            #print("intersection")
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n, path=path)
            return s1, -1.0, done

        # If the airfoil thickness distribution is too spikey, return no reward and the new airfoil
        # Too spikey is defined as having more than a 1 peak
        elif n_peaks_thickness >= 2:
            #print("thickness peaks")
            # Update the stored airfoil
            self.surface_y = s2
            self.surface_normal = surface_normal_new
        
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n, path=path)
            return s2.reshape(2 * self.n_panels_per_surface + 1), 0.0, done
        
        # If the airfoil has a turning angle that is too great (>81.4 deg), return no reward and the new airfoil
        elif (new_turning_angles < 0.15).any():
            #print("turning angles")
            # Update the stored airfoil
            self.surface_y = s2
            self.surface_normal = surface_normal_new
            
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n, path=path)
            return s2.reshape(2 * self.n_panels_per_surface + 1), 0.0, done
        
        # If the airfoil is too thin, return no reward and the new airfoil
        elif (maximum_thickness < 0.05):
            #print("too thin")
            # Update the stored airfoil
            self.surface_y = s2
            self.surface_normal = surface_normal_new
            
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n, path=path)
            return s2.reshape(2 * self.n_panels_per_surface + 1), 0.0, done
        
        # If the airfoil is too thick, return no reward and the new airfoil
        elif (maximum_thickness > 0.25):
            #print("too thick")
            # Update the stored airfoil
            self.surface_y = s2
            self.surface_normal = surface_normal_new
            
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n, path=path)
            return s2.reshape(2 * self.n_panels_per_surface + 1), 0.0, done
        
        # If the point of maximum thickness is too far back, return no reward and the new airfoil
        elif (maximum_thickness_location > 0.25):
            #print("max thickness too far back")
            # Update the stored airfoil
            self.surface_y = s2
            self.surface_normal = surface_normal_new
            
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n, path=path)
            return s2.reshape(2 * self.n_panels_per_surface + 1), 0.0, done

        # If the point of minimum thickness is not the TE, return no reward and the new airfoil
        elif (minimum_thickness <= 0.02):
            #print("min thickness not at TE")
            # Update the stored airfoil
            self.surface_y = s2
            self.surface_normal = surface_normal_new
            
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n, path=path)
            return s2.reshape(2 * self.n_panels_per_surface + 1), 0.0, done
        
        # If the action is acceptable, return a reward proportional to the mean abs percent error between the new airfoil and the design parameters
        else:
            # Update the stored airfoil
            self.surface_y = s2
            self.surface_normal = surface_normal_new
            
            # init loss sum to 0.0
            cl_loss = 0.0
            cdp_loss = 0.0
            cm4c_loss = 0.0
            
            n_test_points = np.size(self.alpha_test_points)
            cp_state = True
            for test_point in range(n_test_points):
                
                # Get the current velocity vector
                v_inf = np.array([np.cos(self.alpha_test_points[test_point]), np.sin(self.alpha_test_points[test_point])])
                
                # Use the vortex panel method to solve for the airfoil's non-dimensional parameters
                cl, cdp, cm4c, cp_state = self.solve_cl_cdp_cm4c(v_inf)
                if not(cp_state):
                    break
                
                # Update the loss function
                cl_loss += abs((cl - self.cl_test_points[test_point]) / self.cl_test_points[test_point])
                cdp_loss += abs((cdp - self.cdp_test_points[test_point]) / self.cdp_test_points[test_point])
                cm4c_loss += abs((cm4c - self.cm4c_test_points[test_point]) / self.cm4c_test_points[test_point])
                
            # If the cp is bad, return a small positive reward
            if not(cp_state):
                #print("bad cp")
                # Visualize airfoil
                if(vis_foil):
                    self.visualize_airfoil(n, path=path)
                return s2.reshape(2 * self.n_panels_per_surface + 1), 0.10, done
            else:
                #print("in target")
                # Calculate the total weighted loss
                # Adjust weights to get more tuned results
                cl_loss = cl_loss / n_test_points
                cdp_loss = cdp_loss / n_test_points
                cm4c_loss = cm4c_loss / n_test_points
                cl_loss_weight = 2.0
                cdp_loss_weight = 1.0
                cm4c_loss_weight = 1.0
                total_loss = (cl_loss_weight*cl_loss + cdp_loss_weight*cdp_loss + cm4c_loss_weight*cm4c_loss)/(cl_loss_weight + cdp_loss_weight + cm4c_loss_weight)
                total_loss = np.clip(total_loss, 0.0,reward_depreciation)
            
                # Use the loss to get a reward between 0.5 and 5.0
                reward = np.clip((5.0 * (reward_depreciation - total_loss)),0.5,5.0)
            
            # Visualize airfoil
            if(vis_foil):
                self.visualize_airfoil(n, path=path)
            return s2.reshape(2 * self.n_panels_per_surface + 1), reward, done
        
        ####################################################### /REWARD FUNCTION ######################################################
        
    # Resets the environment to a random airfoil and returns the new airfoil
    # @param vis_foil=False - determined whether the new airfoil is saved as an image
    # @param n=0 - the number label of the newairfoil
    # @return the new airfoil y/c coords (x/c not returned due to even spacing)
    def reset(self, vis_foil=False, n=0, path="", symmetric=False):
        self.curr_step = 0
        
        # Create upper surface that is legal
        upper_surface_normal, upper_surface_y = self.make_upper_surface(symmetric)
        
        # Create lower surface that is legal
        lower_surface_normal, lower_surface_y = self.make_lower_surface(symmetric)
     
        # Combine upper and lower surfaces
        self.surface_x = np.append(self.upper_surface_x[:-1], self.lower_surface_x)
        self.surface_y = np.append(upper_surface_y[:-1], lower_surface_y)
        self.surface_normal = np.append(upper_surface_normal, lower_surface_normal, axis=1)
        self.x_cen_panel = ((self.surface_x + np.roll(self.surface_x,-1)) / 2)[self.n_panels_per_surface:2*self.n_panels_per_surface]
        
        # Visualize airfoil
        if(vis_foil):
            self.visualize_airfoil(n, path=path)
        
        return self.surface_y.reshape(2 * self.n_panels_per_surface + 1)
    
    # Visualizes the pressure distribution over the terminal airfoil at all test points and saves the performance results
    def visualize_cp_save_performance(self, path=""):
        
        # Create a performance book
        performance = {
                'alpha': [],
                'cl': [],
                'cdp': [],
                'cm4c': [],
                'cl_design': [],
                'cdp_design': [],
                'cm4c_design': []
                }
        
        # Get the x coords of the control points
        x_coords = ((self.surface_x + np.roll(self.surface_x,-1)) / 2)[:-1].reshape(2*self.n_panels_per_surface)
        
        # Step through all the test points
        for test_point in range(np.size(self.alpha_test_points)):
            
            # Get the distribution and performance parameters
            v_inf = np.array([np.cos(self.alpha_test_points[test_point]), np.sin(self.alpha_test_points[test_point])])
            cp, panel_length, control_point_set = self.solve_cp(v_inf)
            cl, cdp, cm4c, cp_state = self.solve_cl_cdp_cm4c(v_inf)
            
            # Update the performance book
            performance['alpha'].append(self.alpha_test_points[test_point])
            performance['cl'].append(cl)
            performance['cdp'].append(cdp)
            performance['cm4c'].append(cm4c)
            performance['cl_design'].append(self.cl_test_points[test_point])
            performance['cdp_design'].append(self.cdp_test_points[test_point])
            performance['cm4c_design'].append(self.cm4c_test_points[test_point])
            
            # Plot the pressure distribution
            plt.clf()
            if cp_state:
                plt.scatter(x_coords, cp,c='k')
                plt.plot(x_coords, cp,c='k')
            else:
                plt.scatter(x_coords, cp,c='r')
                plt.plot(x_coords, cp,c='r')
            plt.gca().invert_yaxis()
            title_str = "Cp for Airfoil at Test Point " + str(test_point)
            plt.title(title_str)
            plt.xlabel("x/c [unitless]")
            plt.ylabel("Cp [unitless]")
            plt.xlim([0,1])
            fig = plt.gcf()
            fig.set_size_inches(10, 5)
            save_str = path + "cp_distributions/cp_airfoil_" + str(test_point) + ".png"
            plt.savefig(save_str, dpi = 100)
            plt.close()        
            
        # Format data for saving
        performance_data = [{key : value[i] for key, value in performance.items()} 
                           for i in range(np.size(self.alpha_test_points))]
            
        # Save the performance results as .csv
        open_str = path + "results/performance.csv"
        with open(open_str, 'w', newline='') as csvfile:
            csv_columns = ['alpha','cl','cdp','cm4c','cl_design','cdp_design','cm4c_design']
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in performance_data:
                    writer.writerow(data)
            
    
    # Visualizes the shape of the airfoil
    # @param n - number label of airfoil
    def visualize_airfoil(self, n, path=""):
        plt.clf
        plt.scatter(self.surface_x.reshape(2*self.n_panels_per_surface + 1), self.surface_y.reshape(2*self.n_panels_per_surface + 1))
        plt.plot(self.surface_x.reshape(2*self.n_panels_per_surface + 1), self.surface_y.reshape(2*self.n_panels_per_surface + 1))
        title_str = "Airfoil " + str(n)
        plt.title(title_str)
        plt.xlabel("x/c [unitless]")
        plt.ylabel("y/c [unitless]")
        plt.xlim([0,1])
        plt.ylim([-0.5,0.5])
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        save_str = path + "airfoils/airfoil_" + str(n) + ".png"
        plt.savefig(save_str, dpi = 200)
        plt.close()
        
    # Visualizes the shape of the airfoil over a sequence
    # @param n_sequence - number label of airfoil
    def visualize_airfoil_sequence(self, surface_x, surface_y, n_seq, path=""):
        for curr_airfoil in range(len(n_seq)):
            plt.clf
            plt.scatter(surface_x[curr_airfoil].squeeze(), surface_y[curr_airfoil].squeeze())
            plt.plot(surface_x[curr_airfoil].squeeze(), surface_y[curr_airfoil].squeeze())
            title_str = "Airfoil " + str(n_seq[curr_airfoil])
            plt.title(title_str)
            plt.xlabel("x/c [unitless]")
            plt.ylabel("y/c [unitless]")
            plt.xlim([0,1])
            plt.ylim([-0.5,0.5])
            fig = plt.gcf()
            fig.set_size_inches(10, 5)
            save_str = path + "airfoils/airfoil_" + str(n_seq[curr_airfoil]) + ".png"
            plt.savefig(save_str, dpi = 200)
            plt.close()