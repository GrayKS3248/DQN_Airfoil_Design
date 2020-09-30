"""
Created on Mon Sep 28 14:23:01 2020

@author: Grayson Schaer
"""

import numpy as np
import matplotlib.pyplot as plt

class Vortex_Panel_Solver():
    def __init__(self, max_num_steps, n_panels_per_surface):
        
        self.max_num_steps = max_num_steps
        self.n_panels_per_surface = n_panels_per_surface
        self.num_actions = 10
        self.z_dirn = np.array([np.zeros(self.n_panels_per_surface), np.zeros(self.n_panels_per_surface), np.ones(self.n_panels_per_surface)])
        self.precision = 10
        
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
    
    # Gets the normal force coefficient based on the pressure distribution
    # @param V_inf - freestream velocity in form np.array([[Vx],[Vy],[0]])
    # @return the normal force coefficient
    def solve_cn(self, v_inf):
        # Get and split cp into lower and upper
        cp = self.solve_cp(v_inf)
        cp_upper = cp[0:self.n_panels_per_surface][::-1].reshape(self.n_panels_per_surface)
        cp_lower = cp[self.n_panels_per_surface:2*self.n_panels_per_surface].reshape(self.n_panels_per_surface)
        
        # Get and split x/c coords
        x_coords = ((self.surface_x + np.roll(self.surface_x,-1)) / 2)[0][self.n_panels_per_surface:2*self.n_panels_per_surface]
        
        # Solve for cn
        cn = np.trapz(cp_lower - cp_upper, x=x_coords)
        return cn
    
    # Gets the axial force coefficient based on the pressure distribution
    # @param V_inf - freestream velocity in form np.array([[Vx],[Vy],[0]])
    # @return the axial force coefficient
    def solve_ca(self, v_inf):
        # Get and split cp into lower and upper
        cp = self.solve_cp(v_inf)
        cp_upper = cp[0:self.n_panels_per_surface][::-1].reshape(self.n_panels_per_surface)
        cp_lower = cp[self.n_panels_per_surface:2*self.n_panels_per_surface].reshape(self.n_panels_per_surface)
        
        # Get and split x/c coords
        x_coords = ((self.surface_x + np.roll(self.surface_x,-1)) / 2)[0][self.n_panels_per_surface:2*self.n_panels_per_surface]
        
        # Solve for cn
        cn = np.trapz(cp_lower - cp_upper, x=x_coords)
        return cn    
    
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
        plt.savefig(save_str, dpi = 500)
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
        save_str = "airfoil_" + str(n) + ".png"
        plt.savefig(save_str, dpi = 500)
        plt.close()
        
    #
    def reset(self):
        print("Resetting Vortex_Panel_Solver...")
    
    #
    def step(self):
        print("Stepping Vortex_Panel_Solver...")