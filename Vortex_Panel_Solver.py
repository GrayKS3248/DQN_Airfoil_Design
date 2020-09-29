"""
Created on Mon Sep 28 14:23:01 2020

@author: Grayson Schaer
"""

import numpy as np

class Vortex_Panel_Solver():
    def __init__(self, max_num_steps, n_panels_per_surface):
        
        self.max_num_steps = max_num_steps
        self.n_panels_per_surface = n_panels_per_surface
        self.num_actions = 10
        self.z_dirn = np.array([np.zeros(self.n_panels_per_surface), np.zeros(self.n_panels_per_surface), np.ones(self.n_panels_per_surface)])
        
        # Create upper surface
        upper_surface_x = np.linspace(1,0,n_panels_per_surface+1).reshape(1, n_panels_per_surface+1)
        upper_surface_y = np.random.rand(1,n_panels_per_surface+1)
        upper_surface_y[0][0] = 0.01
        upper_surface_y[0][-1] = 0.0
        upper_surface_normal = self.get_normal(upper_surface_x, upper_surface_y)
        
        # Create lower surface
        lower_surface_x = np.linspace(0,1,n_panels_per_surface+1).reshape(1, n_panels_per_surface+1)
        lower_surface_y = -1.0 * np.random.rand(1,n_panels_per_surface+1)
        lower_surface_y[0][0] = 0.0
        lower_surface_y[0][-1] = -0.01
        lower_surface_normal = self.get_normal(lower_surface_x, lower_surface_y)
     
        # Combine upper and lower surfaces
        self.surface_x = np.append(upper_surface_x[:,:-1], lower_surface_x).reshape(1, 2 * n_panels_per_surface + 1)
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
    # @return integral in form v_induced', where v_induced' := v_induced = v_induced' * [gamma_j; gamma_{j+1}]
    def solve_integral(self, pj, pjp1, cp):
        # Panel parameters
        panel_length = np.linalg.norm(pjp1 - pj)
        
        # Parameters used for integration
        precision = 10
        s = np.array([np.linspace(pj[0],pjp1[0],precision).reshape(precision), 
                      np.linspace(pj[1],pjp1[1],precision).reshape(precision), 
                      np.linspace(0,0,precision)])
        s_norm = np.linalg.norm(s,axis=0).reshape(1,precision)
        
        # Calculate the radius to control point and its square norm
        r = cp - s
        r_norm_sq = np.einsum('ij,ij->j', r, r).reshape(1,precision)
        
        # Setup integrals
        den = 2 * np.pi * r_norm_sq
        num1 = (1 - s_norm / panel_length)
        num2 = (s_norm / panel_length)
        r0 = -1.0*r[0].reshape(1,precision)
        r1 = r[1].reshape(1,precision)
        
        # solve integrals
        vx_prime_0 = np.trapz(np.multiply(r1, num1) / den, x=s_norm).item()
        vx_prime_1 = np.trapz(np.multiply(r1, num2) / den, x=s_norm).item()
        vy_prime_0 = np.trapz(np.multiply(r0, num1) / den, x=s_norm).item()
        vy_prime_1 = np.trapz(np.multiply(r0, num2) / den, x=s_norm).item()

        # Combine results
        return np.array([[vx_prime_0],[vy_prime_0], [0.0]]), np.array([[vx_prime_1],[vy_prime_1], [0.0]])
    
    #Solves for the A matrix
    # @return the A matrix
    def get_A(self):
        
        # Init the A matrix
        A = np.zeros((2 * self.n_panels_per_surface + 1, 2 * self.n_panels_per_surface + 1))
        
        # Step through all panels
        for curr_control_point in range(2 * self.n_panels_per_surface):
            
            # Get the control point on the current panel (spatial average of boudning points)
            control_point_x = (self.surface_x[0][curr_control_point] + self.surface_x[0][curr_control_point+1])/2
            control_point_y = (self.surface_y[0][curr_control_point] + self.surface_y[0][curr_control_point+1])/2
            control_point = np.array([[control_point_x],[control_point_y],[0.0]])
            # Gather the normal vector
            control_point_normal = self.surface_normal[:,curr_control_point].reshape(1,3)
            
            # Step through all inducing panels
            for curr_inducing_panel in range(2 * self.n_panels_per_surface):
                
                # Get the bounding points of the inducing panel
                pj = np.array([[self.surface_x[0][curr_inducing_panel]], [self.surface_y[0][curr_inducing_panel]], [0.0]])
                pjp1 = np.array([[self.surface_x[0][curr_inducing_panel+1]], [self.surface_y[0][curr_inducing_panel+1]], [0.0]])
                
                # Solve the integral
                v_induced_prime = self.solve_integral(pj, pjp1, control_point)
                
                # Format and update A
                A[curr_control_point][curr_inducing_panel] += np.matmul(control_point_normal, v_induced_prime[0]).item()
                A[curr_control_point][curr_inducing_panel + 1] += np.matmul(control_point_normal, v_induced_prime[1]).item()
                
        # Apply the kutta condition
        A[2 * self.n_panels_per_surface][0] = 1.0
        A[2 * self.n_panels_per_surface][2 * self.n_panels_per_surface] = 1.0
        
        return A
            
        
    # Solves for the B matrix 
    # @param V_inf - freestream velocity in form np.array([[Vx],[Vy],[0]])
    # @return B matrix
    def get_B(self, V_inf):
        
        # init and populate B
        B = np.matmul(V_inf.reshape(1,3), self.surface_normal).reshape(2 * self.n_panels_per_surface,1)
        
        # Kutta condition
        B = np.append(B, 0.0).reshape(2 * self.n_panels_per_surface + 1, 1)
        
        return B
    
    # Solves for the circulation strength at each panel vertex
    # @param V_inf - the free stream velocity
    # @return the circulation strength at each panel vertex
    def solve_gamma(self, V_inf):
        A = self.get_A()
        B = self.get_B(V_inf)
        return np.linalg.solve(A,B)
    
    # Gets the velocity induced by the freestream and the vortex panels at each control point
    # @param
    # @return
    def solve_cp(self, V_inf):
        gamma = self.solve_gamma(V_inf)
        
    
    #
    def reset(self):
        print("Resetting Vortex_Panel_Solver...")
    
    #
    def step(self):
        print("Stepping Vortex_Panel_Solver...")